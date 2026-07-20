#!/usr/bin/env python3
"""Run fermi-hubbard-4 grid with --cost-fn log_proxy_fidelity and exhaustive bonsai."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/treespile/run_experiment.py"
FHAM = REPO / "benchmark/fham/fermi-hubbard-4.fham"
EXPERIMENT_DIR = REPO / "experiments/fermi-hubbard-4-log-proxy-fidelity"
COST_FN = "log_proxy_fidelity"


def t_label(t: float) -> str:
    if abs(t - round(t)) < 1e-9:
        return f"t{int(round(t))}"
    return "t" + format(t, "g").replace(".", "p")


def main() -> int:
    runs = []
    for n in range(1, 6):
        for i in range(1, 11):
            t = i / 10.0
            out = EXPERIMENT_DIR / f"{t_label(t)}-n{n}"
            runs.append((t, n, out))

    todo = [(t, n, out) for t, n, out in runs if not (out / "metrics.json").is_file()]
    print(f"Total: {len(runs)} | done: {len(runs) - len(todo)} | todo: {len(todo)}", flush=True)
    print(f"cost-fn: {COST_FN}", flush=True)
    print("exhaustive: True", flush=True)
    print(f"output:  {EXPERIMENT_DIR}", flush=True)

    failed: list[tuple[float, int, int]] = []
    for idx, (t, n, out) in enumerate(todo, 1):
        print(f"\n===== [{idx}/{len(todo)}] t={t} n={n} -> {out.name} =====", flush=True)
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
            COST_FN,
            "--exhaustive",
        ]
        start = time.time()
        result = subprocess.run(cmd, cwd=REPO)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"FAILED t={t} n={n} (exit {result.returncode}, {elapsed:.1f}s)", flush=True)
            failed.append((t, n, result.returncode))
        else:
            print(f"OK t={t} n={n} ({elapsed:.1f}s)", flush=True)

    print(f"\nDone. Failed: {len(failed)}", flush=True)
    for t, n, code in failed:
        print(f"  t={t} n={n} exit={code}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
