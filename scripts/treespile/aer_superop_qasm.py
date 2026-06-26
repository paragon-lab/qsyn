#!/usr/bin/env python3
"""
Noisy circuit simulation with Qiskit Aer ``superop`` method.

Uses parent ``scripts/`` helpers for IBM/fake backends and sliced calibration JSON.

Examples:
  python aer_superop_qasm.py QC_T.qasm --backend fake_torino -o S_noisy.npy
  python aer_superop_qasm.py QC_T.qasm --ibmq-calibration device.json -o S_noisy.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.library import save_superop
from qiskit_aer.noise import NoiseModel

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from get_backend import get_fake_backend, get_real_backend  # noqa: E402
from ibmq_sliced_backend import (  # noqa: E402
    load_calibration,
    load_sliced_backend,
    noise_model_from_calibration,
)
from qasm_utils import DEFAULT_MAX_QUBITS, load_circuit
from simulate_ideal_vs_noisy_fidelity import (  # noqa: E402
    circuit_to_active_subcircuit,
    get_active_qubit_indices,
    get_noise_model_from_backend,
    noise_model_for_active_qubits,
)

DEFAULT_MAX_QUBITS_SUPEROP = DEFAULT_MAX_QUBITS


def resolve_backend(name: str):
    if name.startswith("fake_"):
        return get_fake_backend(name, verbose=True)
    return get_real_backend(name, verbose=True)


def backend_num_qubits(backend) -> int | None:
    config = getattr(backend, "configuration", lambda: None)()
    if config is not None:
        return getattr(config, "n_qubits", None)
    return getattr(backend, "num_qubits", None)


def prepare_subdevice(
    circuit: QuantumCircuit,
    backend_name: str | None,
    calibration_path: str | Path | None = None,
) -> tuple[QuantumCircuit, list[int], int, NoiseModel | None]:
    active_indices = get_active_qubit_indices(circuit)
    if not active_indices:
        raise ValueError("circuit has no gate operations on any qubit")

    n_register = circuit.num_qubits
    if len(active_indices) < n_register:
        circuit = circuit_to_active_subcircuit(circuit, active_indices)

    if backend_name is None and calibration_path is None:
        raise ValueError("noisy simulation requires --backend or --ibmq-calibration")

    if calibration_path is not None:
        bundle = load_calibration(calibration_path)
        if max(active_indices) >= bundle.num_qubits:
            raise ValueError(
                f"circuit uses logical qubit indices up to {max(active_indices)}, "
                f"but calibration export has only {bundle.num_qubits} qubits"
            )
        backend = load_sliced_backend(calibration_path)
        full_noise = noise_model_from_calibration(calibration_path)
        sub_noise = noise_model_for_active_qubits(full_noise, active_indices)
        return circuit, active_indices, n_register, sub_noise

    backend = resolve_backend(backend_name)
    n_backend = backend_num_qubits(backend)
    if n_backend is not None and max(active_indices) >= n_backend:
        raise ValueError(
            f"circuit uses backend qubit indices up to {max(active_indices)}, "
            f"but {backend.name} has only {n_backend} qubits"
        )
    full_noise = get_noise_model_from_backend(backend)
    sub_noise = noise_model_for_active_qubits(full_noise, active_indices)
    return circuit, active_indices, n_register, sub_noise


def simulate_superop(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    label: str = "superop",
) -> np.ndarray:
    qc = circuit.remove_final_measurements(inplace=False)
    save_superop(qc, label=label)

    sim = AerSimulator(method="superop", noise_model=noise_model)
    result = sim.run(qc, shots=1).result()
    data = result.data(0)
    if label not in data:
        raise KeyError(
            f"Result missing key {label!r}; available keys: {list(data.keys())}"
        )
    return np.asarray(data[label], dtype=complex)


def simulate_circuit_file(
    circuit_path: Path,
    backend_name: str | None = None,
    calibration_path: Path | None = None,
    max_qubits: int = DEFAULT_MAX_QUBITS_SUPEROP,
    label: str = "superop",
) -> np.ndarray:
    circuit = load_circuit(circuit_path)
    if circuit.num_qubits == 0:
        raise ValueError("circuit has no qubits")

    circuit, active_indices, _n_register, noise_model = prepare_subdevice(
        circuit, backend_name, calibration_path
    )
    n_qubits = circuit.num_qubits
    if n_qubits > max_qubits:
        dim = 4**n_qubits
        raise ValueError(
            f"{n_qubits} active qubits implies a {dim}×{dim} superoperator "
            f"(>{max_qubits} qubits limit)"
        )

    return simulate_superop(circuit, noise_model, label=label)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Noisy circuit simulation with Qiskit Aer (superop method)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("circuit", type=Path, help="input .qasm circuit")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="IBM backend or fake_* device (uses .env IBMQ_API_KEY for real backends)",
    )
    parser.add_argument(
        "--ibmq-calibration",
        type=Path,
        default=None,
        help="qsyn device write --ibmq JSON (logical qubits 0..n-1)",
    )
    parser.add_argument("-o", "--output", type=Path, help="write superoperator .npy")
    parser.add_argument("--label", type=str, default="superop")
    parser.add_argument("--max-qubits", type=int, default=DEFAULT_MAX_QUBITS_SUPEROP)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.circuit.is_file():
        print(f"Error: circuit file not found: {args.circuit}", file=sys.stderr)
        return 1

    if args.backend is not None and args.ibmq_calibration is not None:
        print("Error: use only one of --backend and --ibmq-calibration", file=sys.stderr)
        return 1

    try:
        superop = simulate_circuit_file(
            args.circuit,
            backend_name=args.backend,
            calibration_path=args.ibmq_calibration,
            max_qubits=args.max_qubits,
            label=args.label,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Superoperator shape: {superop.shape}")

    if args.output is not None:
        np.save(args.output, superop)
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
