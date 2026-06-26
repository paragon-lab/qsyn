#!/usr/bin/env python3
"""
Ideal (noiseless) circuit simulation with Qiskit Aer ``unitary`` method.

Examples:
  python aer_unitary_qasm.py QC_T.qasm -o U_trot.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.library import save_unitary

from qasm_utils import DEFAULT_MAX_QUBITS, load_circuit, prepare_circuit


def simulate_unitary(circuit, label: str = "unitary") -> np.ndarray:
    """Run circuit on Aer (unitary) and return the (2^n, 2^n) matrix."""
    qc = circuit.remove_final_measurements(inplace=False)
    save_unitary(qc, label=label)

    sim = AerSimulator(method="unitary")
    result = sim.run(qc, shots=1).result()
    data = result.data(0)
    if label not in data:
        raise KeyError(
            f"Result missing key {label!r}; available keys: {list(data.keys())}"
        )
    return np.asarray(data[label], dtype=complex)


def simulate_circuit_file(
    circuit_path: Path,
    max_qubits: int = DEFAULT_MAX_QUBITS,
    label: str = "unitary",
) -> np.ndarray:
    """Load, prepare, and simulate a QASM circuit; return unitary matrix."""
    circuit = load_circuit(circuit_path)
    if circuit.num_qubits == 0:
        raise ValueError("circuit has no qubits")

    circuit, active_indices, _n_register = prepare_circuit(circuit)
    n_qubits = circuit.num_qubits
    if n_qubits > max_qubits:
        dim = 2**n_qubits
        raise ValueError(
            f"{n_qubits} active qubits implies a {dim}×{dim} unitary "
            f"(>{max_qubits} qubits limit)"
        )

    return simulate_unitary(circuit, label=label)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ideal circuit simulation with Qiskit Aer (unitary method)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("circuit", type=Path, help="input .qasm circuit")
    parser.add_argument("-o", "--output", type=Path, help="write unitary .npy")
    parser.add_argument("--label", type=str, default="unitary")
    parser.add_argument("--max-qubits", type=int, default=DEFAULT_MAX_QUBITS)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.circuit.is_file():
        print(f"Error: circuit file not found: {args.circuit}", file=sys.stderr)
        return 1

    try:
        circuit = load_circuit(args.circuit)
        circuit, active_indices, n_register = prepare_circuit(circuit)
        unitary = simulate_unitary(circuit, label=args.label)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    n_qubits = circuit.num_qubits
    if n_qubits > args.max_qubits:
        print(
            f"Error: {n_qubits} active qubits exceeds --max-qubits {args.max_qubits}",
            file=sys.stderr,
        )
        return 1

    dim = unitary.shape[0]
    if args.verbose:
        identity = np.eye(dim, dtype=complex)
        unitarity_error = np.max(np.abs(unitary.conj().T @ unitary - identity))
        print(f"Register qubits: {n_register}")
        print(f"Active qubits: {n_qubits} (indices {active_indices})")
        print(f"Unitary dimension: {dim}×{dim}")
        print(f"Max unitarity error ||U†U - I||_∞: {unitarity_error:.3e}")

    if args.output is not None:
        np.save(args.output, unitary)
        print(f"Wrote {dim}×{dim} unitary to {args.output}")
    elif not args.verbose:
        print(f"Computed ideal unitary ({dim}×{dim})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
