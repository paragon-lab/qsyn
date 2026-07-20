#!/usr/bin/env python3
"""
Backfill ESP / proxy fidelity into existing experiment metrics.json files.

Reuses QC_T.qasm + device.json; does not re-run unitary or noisy simulations.

Example:
  python scripts/treespile/backfill_reliability.py experiments/fermi-hubbard-4
  python scripts/treespile/backfill_reliability.py experiments/fermi-hubbard-4 --plot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qasm_utils import REPO_ROOT, find_qsyn_binary
from run_experiment import measure_reliability


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="directory containing per-run subdirs with metrics.json",
    )
    parser.add_argument(
        "--qsyn",
        type=Path,
        default=None,
        help="path to qsyn executable (default: build/qsyn or PATH)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute even when proxy_fidelity is already present",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="regenerate fidelity plots after backfill",
    )
    return parser.parse_args(argv)


def find_run_dirs(experiment_dir: Path) -> list[Path]:
    runs = []
    for metrics_path in sorted(experiment_dir.glob("*/metrics.json")):
        runs.append(metrics_path.parent)
    return runs


def backfill_run(qsyn_path: Path, run_dir: Path, *, force: bool) -> bool:
    metrics_path = run_dir / "metrics.json"
    device_json = run_dir / "device.json"
    qcir_path = run_dir / "QC_T.qasm"

    data = json.loads(metrics_path.read_text())
    if not force and data.get("proxy_fidelity") is not None:
        print(f"  skip {run_dir.name} (proxy_fidelity already set)")
        return False

    if not device_json.is_file():
        raise FileNotFoundError(f"missing {device_json}")
    if not qcir_path.is_file():
        raise FileNotFoundError(f"missing {qcir_path}")

    print(f"  measure {run_dir.name}")
    esp, proxy_fidelity = measure_reliability(qsyn_path, device_json, qcir_path)
    data["esp"] = esp
    data["proxy_fidelity"] = proxy_fidelity
    metrics_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"    esp={esp:.6g}  proxy_fidelity={proxy_fidelity:.6g}")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_dir = args.experiment_dir.expanduser().resolve()
    if not experiment_dir.is_dir():
        print(f"Error: not a directory: {experiment_dir}", file=sys.stderr)
        return 1

    try:
        qsyn_path = args.qsyn if args.qsyn is not None else find_qsyn_binary()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    runs = find_run_dirs(experiment_dir)
    if not runs:
        print(f"Error: no metrics.json under {experiment_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(runs)} runs under {experiment_dir}")
    updated = 0
    failed = 0
    for run_dir in runs:
        try:
            if backfill_run(qsyn_path, run_dir, force=args.force):
                updated += 1
        except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
            failed += 1
            print(f"  FAILED {run_dir.name}: {exc}", file=sys.stderr)

    print(f"Updated {updated}/{len(runs)} (failed: {failed})")

    if args.plot:
        plot_script = SCRIPT_DIR / "plot_trotter_fidelity.py"
        pattern = f"{experiment_dir.relative_to(REPO_ROOT)}/t*-n*/metrics.json"
        cmd = [
            sys.executable,
            str(plot_script),
            "--pattern",
            pattern,
            "--output-dir",
            str(experiment_dir),
            "--metric",
            "noise",
        ]
        print("+", " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            return result.returncode

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
