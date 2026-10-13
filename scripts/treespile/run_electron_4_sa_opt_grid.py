#!/usr/bin/env python3
"""Run electron-4 SA optimization grid: t in 0.1..1.0, n in 1..10.

Methods (sequential):
  1. Pauli weight          (--optimize-level 1, --cost-fn default)
  2. Uniform APSP          (--optimize-level 2, --cost-fn default)
  3. Log proxy fidelity    (--optimize-level 2, --cost-fn log_proxy_fidelity)

Supports --replicates N: each (t,n) is run N times under rep0..rep{N-1}, then
averaged into metrics.json (mean ± sample std).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/treespile/run_experiment.py"
FHAM = REPO / "benchmark/fham/electron-4.fham"

METHODS: list[dict[str, object]] = [
    {
        "name": "pauli-weight",
        "dir": REPO / "experiments/electron-4-sa-pauli-weight",
        "optimize_level": 1,
        "cost_fn": "default",
    },
    {
        "name": "uniform-apsp",
        "dir": REPO / "experiments/electron-4-sa-uniform-apsp",
        "optimize_level": 2,
        "cost_fn": "default",
    },
    {
        "name": "log-proxy-fidelity",
        "dir": REPO / "experiments/electron-4-sa-log-proxy-fidelity",
        "optimize_level": 2,
        "cost_fn": "log_proxy_fidelity",
    },
]
METHOD_BY_NAME = {str(m["name"]): m for m in METHODS}

AVG_FIELDS = (
    "trotterization_fidelity",
    "noise_fidelity",
    "total_fidelity",
    "esp",
    "proxy_fidelity",
    "synthesis_runtime_sec",
    "experiment_runtime_sec",
)

ARTIFACT_NAMES = (
    "H_jw.txt",
    "H_T.txt",
    "QC_T.qasm",
    "QC_T_untranspiled.qasm",
    "C.qasm",
    "U_jw.npy",
    "U_trot.npy",
    "S_noisy.npy",
    "device.json",
    "metrics.json",
)


def t_label(t: float) -> str:
    if abs(t - round(t)) < 1e-9:
        return f"t{int(round(t))}"
    return "t" + format(t, "g").replace(".", "p")


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def migrate_legacy_run_to_rep0(out: Path) -> bool:
    """If a pre-replicate run lives at out/, move its artifacts into out/rep0/."""
    rep0 = out / "rep0"
    if (rep0 / "metrics.json").is_file():
        return False
    # Flat single-run dir: either metrics or QC_T at top level.
    legacy_metrics = out / "metrics.json"
    legacy_qc = out / "QC_T.qasm"
    if not legacy_metrics.is_file() and not legacy_qc.is_file():
        return False
    rep0.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        src = out / name
        if src.is_file():
            shutil.move(str(src), str(rep0 / name))
    print(f"Migrated legacy run -> {rep0}", flush=True)
    return True


def aggregate_replicates(out: Path, replicates: int) -> dict:
    reps = []
    for k in range(replicates):
        path = out / f"rep{k}" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing replicate metrics: {path}")
        reps.append(json.loads(path.read_text()))

    base = dict(reps[0])
    base["n_replicates"] = replicates
    base["replicate_dirs"] = [f"rep{k}" for k in range(replicates)]
    # Prefer parent-level artifact pointers for convenience.
    base["artifacts"] = {
        "metrics": str(out / "metrics.json"),
        "replicates": [str(out / f"rep{k}") for k in range(replicates)],
    }
    for field in AVG_FIELDS:
        vals = [float(r[field]) for r in reps if r.get(field) is not None]
        if not vals:
            base[field] = None
            base[f"{field}_std"] = None
            continue
        mean, std = mean_std(vals)
        base[field] = mean
        base[f"{field}_std"] = std
    # Keep scalar metadata from first rep; mark aggregated.
    base["aggregated"] = True
    return base


def run_one(
    *,
    name: str,
    t: float,
    n: int,
    out: Path,
    opt: int,
    cost_fn: str,
    seed: int | None = None,
) -> int:
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(FHAM),
        "-t",
        str(t),
        "-n",
        str(n),
        "-o",
        str(out),
        "--fake-device",
        "--device",
        "torino",
        "--cost-fn",
        cost_fn,
        "--optimize-level",
        str(opt),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    start = time.time()
    result = subprocess.run(cmd, cwd=REPO)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(
            f"FAILED {name} t={t} n={n} out={out} "
            f"(exit {result.returncode}, {elapsed:.1f}s)",
            flush=True,
        )
    else:
        # Record wall time of the full experiment if synthesis-only is already set.
        metrics_path = out / "metrics.json"
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text())
            payload.setdefault("experiment_runtime_sec", elapsed)
            if payload.get("experiment_runtime_sec") is None:
                payload["experiment_runtime_sec"] = elapsed
            metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"OK {name} t={t} n={n} out={out.name} ({elapsed:.1f}s)", flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHOD_BY_NAME),
        default=sorted(METHOD_BY_NAME),
        help="subset of SA methods to run (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild aggregated metrics.json even if it exists",
    )
    parser.add_argument(
        "--force-replicates",
        action="store_true",
        help="delete and rerun all replicate directories (implies --force)",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="independent SA runs per (t,n); average into metrics.json (default: 1)",
    )
    parser.add_argument(
        "--reuse-legacy",
        action="store_true",
        help="move an existing flat run into rep0 before filling remaining replicates",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help=(
            "if set, each replicate k uses seed = base_seed + 1000*job_index + k "
            "(deterministic SA across reruns)"
        ),
    )
    args = parser.parse_args()
    if args.replicates < 1:
        raise SystemExit("--replicates must be >= 1")
    if args.force_replicates:
        args.force = True

    selected = [METHOD_BY_NAME[name] for name in args.methods]
    jobs: list[tuple[str, float, int, Path, int, str]] = []
    for method in selected:
        name = str(method["name"])
        root = Path(method["dir"])  # type: ignore[arg-type]
        opt = int(method["optimize_level"])  # type: ignore[arg-type]
        cost_fn = str(method["cost_fn"])
        for n in range(1, 11):
            for i in range(1, 11):
                t = i / 10.0
                out = root / f"{t_label(t)}-n{n}"
                jobs.append((name, t, n, out, opt, cost_fn))

    # Migrate before deleting aggregated metrics so legacy runs become rep0.
    if args.reuse_legacy:
        migrated = 0
        for _name, _t, _n, out, _opt, _cost_fn in jobs:
            if migrate_legacy_run_to_rep0(out):
                migrated += 1
        print(f"Reuse-legacy: migrated {migrated} flat runs into rep0/", flush=True)

    if args.force:
        removed = 0
        for _name, _t, _n, out, _opt, _cost_fn in jobs:
            metrics = out / "metrics.json"
            if metrics.is_file():
                metrics.unlink()
                removed += 1
        print(f"Force: removed {removed} aggregated metrics.json files", flush=True)

    print(
        f"Jobs: {len(jobs)} | replicates/job: {args.replicates} | "
        f"methods: {[m['name'] for m in selected]}",
        flush=True,
    )

    failed: list[tuple[str, float, int, int]] = []
    for idx, (name, t, n, out, opt, cost_fn) in enumerate(jobs, 1):
        out.mkdir(parents=True, exist_ok=True)

        if (out / "metrics.json").is_file() and not args.force:
            print(f"[{idx}/{len(jobs)}] skip {name} t={t} n={n} (aggregated exists)", flush=True)
            continue

        print(f"\n===== [{idx}/{len(jobs)}] {name} t={t} n={n} reps={args.replicates} =====", flush=True)
        job_failed = False
        for k in range(args.replicates):
            rep_dir = out / f"rep{k}"
            if (rep_dir / "metrics.json").is_file() and not args.force_replicates:
                print(f"  skip existing {rep_dir.name}", flush=True)
                continue
            if args.force_replicates and rep_dir.is_dir():
                shutil.rmtree(rep_dir)
            seed = None
            if args.base_seed is not None:
                seed = int(args.base_seed) + 1000 * (idx - 1) + k
            rc = run_one(
                name=name, t=t, n=n, out=rep_dir, opt=opt, cost_fn=cost_fn, seed=seed
            )
            if rc != 0:
                failed.append((name, t, n, rc))
                job_failed = True
                break
        if job_failed:
            continue

        try:
            aggregated = aggregate_replicates(out, args.replicates)
        except FileNotFoundError as exc:
            print(f"FAIL aggregate {name} t={t} n={n}: {exc}", flush=True)
            failed.append((name, t, n, 1))
            continue
        (out / "metrics.json").write_text(json.dumps(aggregated, indent=2) + "\n")
        print(
            f"Aggregated {name} t={t} n={n}: "
            f"F_trot={aggregated['trotterization_fidelity']:.6f}±"
            f"{aggregated.get('trotterization_fidelity_std') or 0:.6f} "
            f"F_noise={aggregated['noise_fidelity']:.6f}±"
            f"{aggregated.get('noise_fidelity_std') or 0:.6f}",
            flush=True,
        )

    print(f"\nDone. Failed: {len(failed)}", flush=True)
    for name, t, n, code in failed:
        print(f"  {name} t={t} n={n} exit={code}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
