#!/usr/bin/env python3
"""
Build U(t) = exp(-i H t) from a qbham text file (``qbham read`` / ``qbham write`` format).

Examples:
  python qbham_evolution.py H_jw.txt -t 0.1 -o U_jw.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp
from scipy.linalg import expm

from qasm_utils import DEFAULT_MAX_QUBITS


def qsyn_pauli_to_qiskit(pauli: str) -> Pauli:
    """Map qsyn Pauli labels (qubit 0 = leftmost) to Qiskit / OpenQASM ordering."""
    return Pauli(pauli[::-1])


def read_qbham(path: Path) -> list[tuple[float, str]]:
    """Parse a qbham Hamiltonian file into (coefficient, pauli_string) pairs."""
    terms: list[tuple[float, str]] = []
    expected_n_qubits: int | None = None

    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(
                    f'{path}:{lineno}: expected "coefficient PauliString", got {raw!r}'
                )

            coeff = float(parts[0])
            pauli = parts[1].strip().upper()
            if not pauli:
                raise ValueError(f"{path}:{lineno}: empty Pauli string")

            for char in pauli:
                if char not in "IXYZ":
                    raise ValueError(
                        f"{path}:{lineno}: invalid Pauli character {char!r} in {pauli!r}"
                    )

            if expected_n_qubits is None:
                expected_n_qubits = len(pauli)
            elif len(pauli) != expected_n_qubits:
                raise ValueError(
                    f"{path}:{lineno}: inconsistent qubit count "
                    f"(expected {expected_n_qubits}, got {len(pauli)} in {pauli!r})"
                )

            terms.append((coeff, pauli))

    if not terms:
        raise ValueError(f"{path}: no Hamiltonian terms found")

    return terms


def hamiltonian_matrix(terms: list[tuple[float, str]]) -> np.ndarray:
    """Assemble H = sum_i c_i P_i as a dense matrix.

    ``terms`` use qsyn/qbham Pauli labels (qubit 0 = leftmost character). The
    returned matrix follows Qiskit / OpenQASM basis ordering
    (|q_{n-1} ... q_0>, qubit 0 = LSB) so it matches Clifford and Trotter
    circuits from qsyn.
    """
    pauli_list = [(qsyn_pauli_to_qiskit(pauli).to_label(), coeff) for coeff, pauli in terms]
    hamiltonian = np.asarray(SparsePauliOp.from_list(pauli_list).to_matrix(), dtype=complex)

    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-10, rtol=1e-10):
        max_skew = np.max(np.abs(hamiltonian - hamiltonian.conj().T))
        raise ValueError(f"Hamiltonian is not Hermitian (max skew = {max_skew:.3g})")

    return hamiltonian


def evolution_operator(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    """Return U(t) = exp(-i H t)."""
    return expm(-1j * time * hamiltonian)


def compute_evolution(hamiltonian_path: Path, time: float, max_qubits: int = DEFAULT_MAX_QUBITS) -> np.ndarray:
    """Read qbham file and return U(t)."""
    terms = read_qbham(hamiltonian_path)
    n_qubits = len(terms[0][1])
    if n_qubits > max_qubits:
        raise ValueError(
            f"Hamiltonian has {n_qubits} qubits (limit {max_qubits}). "
            "Dense exp(-iHt) is impractical beyond this limit."
        )
    hamiltonian = hamiltonian_matrix(terms)
    return evolution_operator(hamiltonian, time)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute U(t) = exp(-i H t) from a qbham Hamiltonian file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="qbham text file")
    parser.add_argument("-t", "--time", type=float, required=True, help="evolution time")
    parser.add_argument("-o", "--output", type=Path, help="write unitary .npy")
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=DEFAULT_MAX_QUBITS,
        help=f"refuse when n exceeds this (default: {DEFAULT_MAX_QUBITS})",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.is_file():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        terms = read_qbham(args.input)
        unitary = compute_evolution(args.input, args.time, args.max_qubits)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    n_qubits = len(terms[0][1])
    dim = unitary.shape[0]

    if args.verbose:
        identity = np.eye(dim, dtype=complex)
        unitarity_error = np.max(np.abs(unitary.conj().T @ unitary - identity))
        print(f"Qubits: {n_qubits}")
        print(f"Hilbert space dimension: {dim}")
        print(f"Terms: {len(terms)}")
        print(f"Evolution time: {args.time}")
        print(f"Max unitarity error ||U†U - I||_∞: {unitarity_error:.3e}")

    if args.output is not None:
        np.save(args.output, unitary)
        print(f"Wrote {dim}x{dim} unitary to {args.output}")
    elif not args.verbose:
        print(
            f"Computed U(t) for {n_qubits} qubits ({dim}x{dim}); "
            "use -o to save or --verbose for details."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
