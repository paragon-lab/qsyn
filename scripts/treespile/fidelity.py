"""
Process fidelity utilities for the treespile encoding experiment.

Compares a reference unitary U_ideal to a decoded implementation channel:

    F_proc = (1/d²) Σ_k |Tr(U_ideal† K_k')|²

where K_k' = C† K_k C for Kraus operators of the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from qiskit.quantum_info import Kraus, Operator, SuperOp, process_fidelity, state_fidelity, Statevector

from qasm_utils import load_circuit

ImplementationKind = Literal["unitary", "superop"]


@dataclass(frozen=True)
class ExperimentMetrics:
    """Decoded process fidelities for the encoding experiment."""

    trotterization: float
    total: float | None
    noise: float | None
    n_qubits: int
    kraus_rank_trotter: int
    kraus_rank_noisy: int | None


def load_ideal_unitary(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"ideal file not found: {path}")
    matrix = np.load(path)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"ideal must be square 2-D, got shape {matrix.shape}")
    return np.asarray(matrix, dtype=complex)


def classify_implementation(side: int, hilbert_dim: int) -> ImplementationKind:
    if side == hilbert_dim:
        return "unitary"
    if side == hilbert_dim * hilbert_dim:
        return "superop"
    raise ValueError(
        f"matrix side {side} does not match unitary ({hilbert_dim}) "
        f"or superoperator ({hilbert_dim * hilbert_dim})"
    )


def load_implementation(path: Path, hilbert_dim: int) -> tuple[ImplementationKind, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"implemented file not found: {path}")
    matrix = np.load(path)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"implemented must be square 2-D, got shape {matrix.shape}")
    kind = classify_implementation(matrix.shape[0], hilbert_dim)
    return kind, np.asarray(matrix, dtype=complex)


def clifford_unitary(path: Path) -> np.ndarray:
    circuit = load_circuit(path).remove_final_measurements(inplace=False)
    if circuit.num_qubits == 0:
        raise ValueError(f"Clifford circuit has no qubits: {path}")
    if len(circuit.data) == 0:
        raise ValueError(f"Clifford circuit has no gates: {path}")
    return np.asarray(Operator(circuit), dtype=complex)


def kraus_matrices_from_superop(superop: np.ndarray) -> list[np.ndarray]:
    kraus = Kraus(SuperOp(superop))
    data = np.asarray(kraus.data, dtype=complex)
    if data.ndim == 2:
        return [data]
    return [data[i] for i in range(data.shape[0])]


def kraus_from_implementation(kind: ImplementationKind, matrix: np.ndarray) -> list[np.ndarray]:
    if kind == "unitary":
        return [matrix]
    return kraus_matrices_from_superop(matrix)


def decode_kraus(kraus_ops: list[np.ndarray], clifford: np.ndarray) -> list[np.ndarray]:
    clifford_dagger = clifford.conj().T
    return [clifford_dagger @ kraus @ clifford for kraus in kraus_ops]


def process_fidelity_unitary_kraus(reference: np.ndarray, kraus_ops: list[np.ndarray]) -> float:
    dim = reference.shape[0]
    reference_dagger = reference.conj().T
    total = sum(abs(np.trace(reference_dagger @ kraus)) ** 2 for kraus in kraus_ops)
    return float(total / (dim * dim))


def process_fidelity_channels(
    reference_kraus: list[np.ndarray],
    candidate_kraus: list[np.ndarray],
) -> float:
    """Process fidelity between two quantum channels (Qiskit Kraus convention)."""
    return float(process_fidelity(Kraus(candidate_kraus), Kraus(reference_kraus)))


def check_unitary(matrix: np.ndarray, label: str, atol: float = 1e-8) -> float:
    dim = matrix.shape[0]
    identity = np.eye(dim, dtype=complex)
    error = float(np.max(np.abs(matrix.conj().T @ matrix - identity)))
    if error > atol:
        raise ValueError(f"{label} is not unitary within tolerance (error = {error:.3g})")
    return error


def check_cptp(superop: np.ndarray, atol: float = 1e-7) -> None:
    if not Kraus(SuperOp(superop)).is_cptp(atol=atol):
        raise ValueError("superoperator is not CPTP within tolerance")


def compare_to_ideal(
    u_ideal: np.ndarray,
    implemented_path: Path,
    clifford_path: Path,
) -> tuple[float, list[np.ndarray], ImplementationKind]:
    """Return (F_proc, decoded_kraus, implementation_kind)."""
    u_clifford = clifford_unitary(clifford_path)
    if u_ideal.shape != u_clifford.shape:
        raise ValueError(
            f"ideal {u_ideal.shape} and Clifford {u_clifford.shape} dimension mismatch"
        )

    check_unitary(u_ideal, "ideal")
    check_unitary(u_clifford, "clifford")

    impl_kind, impl_matrix = load_implementation(implemented_path, u_clifford.shape[0])
    if impl_kind == "unitary":
        check_unitary(impl_matrix, "implemented")
    else:
        check_cptp(impl_matrix)

    kraus_decoded = decode_kraus(kraus_from_implementation(impl_kind, impl_matrix), u_clifford)
    fidelity = process_fidelity_unitary_kraus(u_ideal, kraus_decoded)
    return fidelity, kraus_decoded, impl_kind


def evaluate_experiment(
    u_ideal_path: Path,
    u_trotter_path: Path,
    clifford_path: Path,
    u_noisy_path: Path | None = None,
) -> ExperimentMetrics:
    """Compute trotterization, noise, and total errors for the experiment."""
    u_ideal = load_ideal_unitary(u_ideal_path)

    f_trot, kraus_trot, _ = compare_to_ideal(u_ideal, u_trotter_path, clifford_path)

    f_total = None
    f_noise = None
    kraus_noisy_rank = None

    if u_noisy_path is not None:
        f_total, kraus_noisy, _ = compare_to_ideal(u_ideal, u_noisy_path, clifford_path)
        f_noise = process_fidelity_channels(kraus_trot, kraus_noisy)
        kraus_noisy_rank = len(kraus_noisy)

    dim = u_ideal.shape[0]
    n_qubits = int(round(np.log2(dim)))

    return ExperimentMetrics(
        trotterization=f_trot,
        total=f_total,
        noise=f_noise,
        n_qubits=n_qubits,
        kraus_rank_trotter=len(kraus_trot),
        kraus_rank_noisy=kraus_noisy_rank,
    )


def format_metrics(metrics: ExperimentMetrics) -> str:
    lines = [
        f"Qubits: {metrics.n_qubits}",
        f"Trotterization fidelity  F(U_jw, C† E_trot C): {metrics.trotterization:.12f}"
        f"  (error {1.0 - metrics.trotterization:.12f})",
    ]
    if metrics.total is not None:
        lines.append(
            f"Total fidelity           F(U_jw, C† E_noisy C): {metrics.total:.12f}"
            f"  (error {1.0 - metrics.total:.12f})"
        )
    if metrics.noise is not None:
        lines.append(
            f"Noise fidelity           F(C† E_trot C, C† E_noisy C): {metrics.noise:.12f}"
            f"  (error {1.0 - metrics.noise:.12f})"
        )
    return "\n".join(lines)


def verbose_diagnostics(
    u_ideal: np.ndarray,
    kraus_decoded: list[np.ndarray],
    clifford_path: Path,
) -> str:
    n_qubits = int(round(np.log2(u_ideal.shape[0])))
    lines = [
        f"Clifford circuit: {clifford_path}",
        f"Kraus rank after decoding: {len(kraus_decoded)}",
    ]
    if len(kraus_decoded) == 1:
        u_dec = kraus_decoded[0]
        qiskit_f = float(process_fidelity(Operator(u_dec), Operator(u_ideal)))
        zero = Statevector.from_label("0" * n_qubits)
        f_state = float(state_fidelity(zero.evolve(u_ideal), zero.evolve(u_dec)))
        lines.append(f"Qiskit process_fidelity (unitary): {qiskit_f:.12f}")
        lines.append(f"|0…0⟩ state fidelity: {f_state:.12f}")
    else:
        qiskit_f = float(process_fidelity(Kraus(kraus_decoded), Operator(u_ideal)))
        lines.append(f"Qiskit process_fidelity (Kraus): {qiskit_f:.12f}")
    return "\n".join(lines)
