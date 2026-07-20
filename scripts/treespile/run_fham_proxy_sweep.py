#!/usr/bin/env python3
"""
Sweep proxy fidelity (no Aer) over all benchmark/fham/*.fham files.

Fixed evolution time; varies Trotter steps and cost function.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QSYN_SCRIPT = REPO / "scripts/treespile/proxy_only.qsyn"
FHAM_DIR = REPO / "benchmark/fham"
OUT_ROOT = REPO / "experiments/fham-proxy-sweep"

_ESP_RE = re.compile(r"Estimated success probability \(ESP\):\s*([\d.eE+-]+)")
_PROXY_RE = re.compile(r"Proxy fidelity:\s*([\d.eE+-]+)")


def find_qsyn() -> Path:
    for cand in (REPO / "build/qsyn", REPO / "qsyn"):
        if cand.is_file() and cand.stat().st_mode & 0o111:
            return cand
    raise FileNotFoundError("qsyn binary not found")


def n_modes(path: Path) -> int:
    modes: set[int] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"\([^)]*\)\s+(.*)", line)
        ops = (m.group(1) if m else line).split()
        for tok in ops:
            modes.add(int(tok[:-1] if tok.endswith("^") else tok))
    return max(modes) if modes else 0


def parse_metrics(text: str) -> tuple[float, float]:
    esp = _ESP_RE.search(text)
    proxy = _PROXY_RE.search(text)
    if not esp or not proxy:
        raise ValueError("failed to parse ESP / proxy fidelity from qsyn output")
    return float(esp.group(1)), float(proxy.group(1))


def run_one(
    qsyn: Path,
    fham: Path,
    out_dir: Path,
    *,
    time_val: float,
    steps: int,
    cost_fn: str,
    exhaustive: bool,
    device: str = "fake_torino",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(qsyn),
        str(QSYN_SCRIPT),
        str(fham.resolve()),
        str(out_dir.resolve()),
        str(time_val),
        str(steps),
        device,
        cost_fn,
        "-e" if exhaustive else "",
    ]
    print("+", " ".join(c for c in cmd if c), flush=True)
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"qsyn failed with exit {result.returncode}")
    esp, proxy = parse_metrics(result.stdout)
    payload = {
        "fham": str(fham.resolve()),
        "benchmark": fham.stem,
        "n_modes": n_modes(fham),
        "time": time_val,
        "trotter_steps": steps,
        "device": device,
        "cost_fn": cost_fn,
        "exhaustive": exhaustive,
        "esp": esp,
        "proxy_fidelity": proxy,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--time", type=float, default=1.0, help="evolution time (default: 1.0)")
    parser.add_argument(
        "-n",
        "--steps",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Trotter step counts (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--cost-fns",
        nargs="+",
        default=["default", "log_proxy_fidelity"],
        help="cost functions to compare",
    )
    parser.add_argument(
        "--exhaustive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override exhaustive bonsai for all cost fns "
        "(default: exhaustive only for non-default costs)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="subset of stems (default: all *.fham)",
    )
    args = parser.parse_args()

    qsyn = find_qsyn()
    fhams = sorted(FHAM_DIR.glob("*.fham"))
    if args.benchmarks:
        wanted = set(args.benchmarks)
        fhams = [p for p in fhams if p.stem in wanted]

    jobs = []
    for fham in fhams:
        for steps in args.steps:
            for cost_fn in args.cost_fns:
                out = OUT_ROOT / fham.stem / f"t{args.time:g}-n{steps}-{cost_fn}"
                jobs.append((fham, steps, cost_fn, out))

    def use_exhaustive(cost_fn: str) -> bool:
        if args.exhaustive is not None:
            return args.exhaustive
        # Uniform/default eccentricity center is better without exhaustive;
        # weighted costs need exhaustive root search.
        return cost_fn != "default"

    todo = [j for j in jobs if not (j[3] / "metrics.json").is_file()]
    print(
        f"Total: {len(jobs)} | done: {len(jobs) - len(todo)} | todo: {len(todo)}",
        flush=True,
    )
    exh_note = (
        f"override={args.exhaustive}"
        if args.exhaustive is not None
        else "per-cost (default=False, others=True)"
    )
    print(
        f"t={args.time}  n={args.steps}  cost_fns={args.cost_fns}  exhaustive={exh_note}",
        flush=True,
    )

    failed: list[str] = []
    for idx, (fham, steps, cost_fn, out) in enumerate(todo, 1):
        exh = use_exhaustive(cost_fn)
        label = f"{fham.stem} n={steps} cost={cost_fn} exh={exh}"
        print(f"\n===== [{idx}/{len(todo)}] {label} =====", flush=True)
        start = time.time()
        try:
            payload = run_one(
                qsyn,
                fham,
                out,
                time_val=args.time,
                steps=steps,
                cost_fn=cost_fn,
                exhaustive=exh,
            )
            print(
                f"OK {label} proxy={payload['proxy_fidelity']:.6g} "
                f"({time.time() - start:.1f}s)",
                flush=True,
            )
        except Exception as exc:
            print(f"FAILED {label}: {exc}", flush=True)
            failed.append(label)

    print(f"\nDone. Failed: {len(failed)}", flush=True)
    for item in failed:
        print(f"  {item}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
