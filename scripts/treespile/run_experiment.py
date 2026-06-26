#!/usr/bin/env python3
"""
Run the treespile encoding experiment end-to-end.

1. qsyn: JW Hamiltonian, treespile circuit, Clifford decoding circuit
2. Python: exact U_jw(t), noiseless trotter unitary, optional noisy superop
3. Python: trotterization / noise / total process fidelities

Examples:
  python run_experiment.py benchmark/fham/electron-4.fham -t 0.1 -n 10 \\
      -o experiments/e4 --fake-device fake_torino

  python run_experiment.py benchmark/fham/electron-4.fham -t 0.1 -n 10 \\
      -o experiments/e4 --device ibm_torino
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aer_superop_qasm import (
    DEFAULT_MAX_QUBITS_SUPEROP,
    simulate_circuit_file as simulate_noisy_superop,
)
from aer_unitary_qasm import simulate_circuit_file as simulate_noiseless_unitary
from fidelity import evaluate_experiment, format_metrics
from qbham_evolution import compute_evolution
from qasm_utils import DEFAULT_MAX_QUBITS, REPO_ROOT, find_qsyn_binary

EXPERIMENT_QSYN = SCRIPT_DIR / "experiment.qsyn"


def run_qsyn(
    qsyn_path: Path,
    fham_path: Path,
    out_dir: Path,
    time: float,
    steps: int,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(qsyn_path),
        str(EXPERIMENT_QSYN),
        str(fham_path.resolve()),
        str(out_dir.resolve()),
        str(time),
        str(steps),
        device,
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the treespile encoding simulation experiment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("fham", type=Path, help="fermionic Hamiltonian input file")
    parser.add_argument("-t", "--time", type=float, required=True, help="evolution time")
    parser.add_argument("-n", "--steps", type=int, required=True, help="Trotter steps")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="directory for artifacts and results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="fake_torino",
        help="device for treespile / noisy sim (default: fake_torino)",
    )
    parser.add_argument(
        "--fake-device",
        action="store_true",
        help="prefix --device with fake_ when calling qsyn device fetch -f",
    )
    parser.add_argument(
        "--qsyn",
        type=Path,
        default=None,
        help="path to qsyn executable (default: build/qsyn or PATH)",
    )
    parser.add_argument(
        "--skip-qsyn",
        action="store_true",
        help="skip qsyn stage; use existing artifacts in --output-dir",
    )
    parser.add_argument(
        "--skip-noisy",
        action="store_true",
        help="skip noisy superoperator simulation",
    )
    parser.add_argument(
        "--ibmq-calibration",
        type=Path,
        default=None,
        help="calibration JSON for noisy sim (default: OUT/device.json if present)",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=DEFAULT_MAX_QUBITS,
        help=f"max qubits for dense U(t) and unitary sim (default: {DEFAULT_MAX_QUBITS})",
    )
    parser.add_argument(
        "--max-superop-qubits",
        type=int,
        default=DEFAULT_MAX_QUBITS_SUPEROP,
        help=(
            "max active qubits for noisy superoperator sim "
            f"(default: {DEFAULT_MAX_QUBITS_SUPEROP}; memory scales as 16·4^(2n) bytes)"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "h_jw": out_dir / "H_jw.txt",
        "qc_t": out_dir / "QC_T.qasm",
        "c": out_dir / "C.qasm",
        "u_jw": out_dir / "U_jw.npy",
        "u_trot": out_dir / "U_trot.npy",
        "s_noisy": out_dir / "S_noisy.npy",
        "device": out_dir / "device.json",
        "metrics": out_dir / "metrics.json",
    }


def require_artifacts(paths: dict[str, Path], need_noisy: bool) -> None:
    required = ["h_jw", "qc_t", "c"]
    for key in required:
        if not paths[key].is_file():
            raise FileNotFoundError(f"missing artifact: {paths[key]}")
    if need_noisy and not paths["s_noisy"].is_file():
        raise FileNotFoundError(f"missing noisy artifact: {paths['s_noisy']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.skip_qsyn and not args.fham.is_file():
        print(f"Error: FHam file not found: {args.fham}", file=sys.stderr)
        return 1

    try:
        qsyn_path = args.qsyn if args.qsyn is not None else find_qsyn_binary()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve()
    paths = artifact_paths(out_dir)

    device_name = args.device
    if args.fake_device and not device_name.startswith("fake_"):
        device_name = f"fake_{device_name}"

    try:
        if not args.skip_qsyn:
            run_qsyn(qsyn_path, args.fham, out_dir, args.time, args.steps, device_name)

        require_artifacts(paths, need_noisy=False)

        calibration_path = args.ibmq_calibration
        if calibration_path is None and paths["device"].is_file():
            calibration_path = paths["device"]

        print(f"\n=== Ground truth U_jw({args.time}) ===")
        u_jw = compute_evolution(paths["h_jw"], args.time, args.max_qubits)
        np.save(paths["u_jw"], u_jw)
        print(f"Wrote {paths['u_jw']} ({u_jw.shape[0]}×{u_jw.shape[0]})")

        print(f"\n=== Noiseless trotter simulation ===")
        u_trot = simulate_noiseless_unitary(
            paths["qc_t"], max_qubits=args.max_qubits
        )
        np.save(paths["u_trot"], u_trot)
        print(f"Wrote {paths['u_trot']} ({u_trot.shape[0]}×{u_trot.shape[0]})")

        noisy_path = None
        if not args.skip_noisy:
            print(f"\n=== Noisy simulation ===")
            if calibration_path is not None and calibration_path.is_file():
                s_noisy = simulate_noisy_superop(
                    paths["qc_t"],
                    calibration_path=calibration_path,
                    max_qubits=args.max_superop_qubits,
                )
            else:
                s_noisy = simulate_noisy_superop(
                    paths["qc_t"],
                    backend_name=device_name,
                    max_qubits=args.max_superop_qubits,
                )
            np.save(paths["s_noisy"], s_noisy)
            noisy_path = paths["s_noisy"]
            print(f"Wrote {paths['s_noisy']} ({s_noisy.shape[0]}×{s_noisy.shape[0]})")

        print(f"\n=== Process fidelities ===")
        metrics = evaluate_experiment(
            paths["u_jw"],
            paths["u_trot"],
            paths["c"],
            noisy_path,
        )
        print(format_metrics(metrics))

        results = {
            "fham": str(args.fham.resolve()),
            "time": args.time,
            "trotter_steps": args.steps,
            "device": device_name,
            "trotterization_fidelity": metrics.trotterization,
            "noise_fidelity": metrics.noise,
            "total_fidelity": metrics.total,
            "n_qubits": metrics.n_qubits,
            "artifacts": {key: str(path) for key, path in paths.items()},
        }
        paths["metrics"].write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nWrote {paths['metrics']}")

        if args.verbose:
            print(json.dumps(results, indent=2))

        return 0

    except subprocess.CalledProcessError as exc:
        print(f"Error: qsyn command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
