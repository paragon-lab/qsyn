#!/usr/bin/env python3
"""Backfill metrics.json for SA electron-4 runs that failed fidelity eval.

Cause: SA can leave a logical qubit idle; Aer previously dropped idle qubits,
yielding 16×16 unitaries vs 32×32 U_jw. Re-simulates with drop_idle=False and
writes metrics.json, including experiment wall time parsed from the grid log.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aer_superop_qasm import simulate_circuit_file as simulate_noisy_superop
from aer_unitary_qasm import simulate_circuit_file as simulate_noiseless_unitary
from fidelity import evaluate_experiment, format_metrics
from qasm_utils import REPO_ROOT, find_qsyn_binary
from run_experiment import measure_reliability

METHODS = {
    "pauli-weight": {
        "dir": REPO_ROOT / "experiments/electron-4-sa-pauli-weight",
        "optimize_level": 1,
        "cost_fn": "default",
    },
    "uniform-apsp": {
        "dir": REPO_ROOT / "experiments/electron-4-sa-uniform-apsp",
        "optimize_level": 2,
        "cost_fn": "default",
    },
    "log-proxy-fidelity": {
        "dir": REPO_ROOT / "experiments/electron-4-sa-log-proxy-fidelity",
        "optimize_level": 2,
        "cost_fn": "log_proxy_fidelity",
    },
}

FAIL_RE = re.compile(
    r"FAILED (?P<method>[\w-]+) t=(?P<t>[0-9.]+) n=(?P<n>\d+) \(exit \d+, (?P<sec>[0-9.]+)s\)"
)
OK_RE = re.compile(
    r"OK (?P<method>[\w-]+) t=(?P<t>[0-9.]+) n=(?P<n>\d+) \((?P<sec>[0-9.]+)s\)"
)


def t_label(t: float) -> str:
    if abs(t - round(t)) < 1e-9:
        return f"t{int(round(t))}"
    return "t" + format(t, "g").replace(".", "p")


def parse_runtimes(log_path: Path) -> dict[tuple[str, float, int], float]:
    text = log_path.read_text()
    out: dict[tuple[str, float, int], float] = {}
    for cre in (FAIL_RE, OK_RE):
        for m in cre.finditer(text):
            key = (m.group("method"), float(m.group("t")), int(m.group("n")))
            out[key] = float(m.group("sec"))
    return out


def backfill_one(
    method: str,
    run_dir: Path,
    *,
    t: float,
    n: int,
    optimize_level: int,
    cost_fn: str,
    wall_sec: float | None,
    qsyn_path: Path,
    force: bool,
) -> str:
    """Return 'ok', 'skip', or raise."""
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        return "skip"

    qc_t = run_dir / "QC_T.qasm"
    c_path = run_dir / "C.qasm"
    h_jw = run_dir / "H_jw.txt"
    u_jw_path = run_dir / "U_jw.npy"
    device = run_dir / "device.json"
    for p in (qc_t, c_path, device):
        if not p.is_file():
            raise FileNotFoundError(f"missing {p}")

    if not u_jw_path.is_file():
        from qbham_evolution import compute_evolution

        u_jw = compute_evolution(h_jw, t, max_qubits=12)
        np.save(u_jw_path, u_jw)
    else:
        u_jw = np.load(u_jw_path)

    dim = int(u_jw.shape[0])
    u_trot_path = run_dir / "U_trot.npy"
    s_noisy_path = run_dir / "S_noisy.npy"

    need_u_trot = True
    if u_trot_path.is_file():
        u_trot = np.load(u_trot_path)
        need_u_trot = u_trot.ndim != 2 or u_trot.shape != (dim, dim)
    if need_u_trot:
        u_trot = simulate_noiseless_unitary(qc_t, drop_idle=False)
        np.save(u_trot_path, u_trot)

    need_s_noisy = True
    if s_noisy_path.is_file():
        s_noisy = np.load(s_noisy_path, mmap_mode="r")
        need_s_noisy = s_noisy.ndim != 2 or s_noisy.shape != (dim * dim, dim * dim)
    if need_s_noisy:
        s_noisy = simulate_noisy_superop(
            qc_t, calibration_path=device, drop_idle=False
        )
        np.save(s_noisy_path, s_noisy)

    metrics = evaluate_experiment(u_jw_path, u_trot_path, c_path, s_noisy_path)
    esp, proxy = measure_reliability(qsyn_path, device, qc_t)

    payload = {
        "fham": str((REPO_ROOT / "benchmark/fham/electron-4.fham").resolve()),
        "time": t,
        "trotter_steps": n,
        "device": "fake_torino",
        "cost_fn": cost_fn,
        "optimize_level": optimize_level,
        "exhaustive": False,
        "synthesis_runtime_sec": wall_sec,
        "experiment_runtime_sec": wall_sec,
        "esp": esp,
        "proxy_fidelity": proxy,
        "trotterization_fidelity": metrics.trotterization,
        "noise_fidelity": metrics.noise,
        "total_fidelity": metrics.total,
        "n_qubits": metrics.n_qubits,
        "backfilled": True,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"OK {method} t={t} n={n}  "
        f"F_trot={metrics.trotterization:.6f} F_noise={metrics.noise:.6f} "
        f"F_tot={metrics.total:.6f} runtime={wall_sec}",
        flush=True,
    )
    print(format_metrics(metrics), flush=True)
    return "ok"


def _worker(job: dict) -> tuple[str, str, float, int]:
    """Picklable worker: returns (status, method, t, n)."""
    import os

    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"
    try:
        status = backfill_one(
            job["method"],
            Path(job["run_dir"]),
            t=job["t"],
            n=job["n"],
            optimize_level=job["optimize_level"],
            cost_fn=job["cost_fn"],
            wall_sec=job["wall_sec"],
            qsyn_path=Path(job["qsyn_path"]),
            force=job["force"],
        )
        return status, job["method"], job["t"], job["n"]
    except Exception as exc:
        print(f"FAIL {job['method']} t={job['t']} n={job['n']}: {exc}", flush=True)
        return "fail", job["method"], job["t"], job["n"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO_ROOT / "experiments/logs/electron-4-sa-opt-20260726-145515.log",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="stop after N backfills (0=all)")
    parser.add_argument("--jobs", type=int, default=2, help="parallel workers (default: 2)")
    args = parser.parse_args()

    runtimes = parse_runtimes(args.log) if args.log.is_file() else {}
    print(f"Parsed {len(runtimes)} runtimes from {args.log}", flush=True)
    qsyn_path = find_qsyn_binary()

    jobs: list[dict] = []
    for method, meta in METHODS.items():
        root: Path = meta["dir"]  # type: ignore[assignment]
        for n in range(1, 11):
            for i in range(1, 11):
                t = i / 10.0
                run_dir = root / f"{t_label(t)}-n{n}"
                if not run_dir.is_dir():
                    continue
                jobs.append(
                    {
                        "method": method,
                        "run_dir": str(run_dir),
                        "t": t,
                        "n": n,
                        "optimize_level": int(meta["optimize_level"]),
                        "cost_fn": str(meta["cost_fn"]),
                        "wall_sec": runtimes.get((method, t, n)),
                        "qsyn_path": str(qsyn_path),
                        "force": args.force,
                    }
                )
                if args.limit and len(jobs) >= args.limit:
                    break
            if args.limit and len(jobs) >= args.limit:
                break
        if args.limit and len(jobs) >= args.limit:
            break

    print(f"Queued {len(jobs)} runs with jobs={args.jobs}", flush=True)
    done = skipped = failed = 0
    if args.jobs <= 1:
        for job in jobs:
            status, _, _, _ = _worker(job)
            if status == "ok":
                done += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_worker, job) for job in jobs]
            for fut in as_completed(futures):
                status, _, _, _ = fut.result()
                if status == "ok":
                    done += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1

    print(f"backfilled={done} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
