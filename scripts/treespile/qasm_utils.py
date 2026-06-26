"""Shared QASM loading utilities for treespile experiment scripts."""

from __future__ import annotations

import sys
from pathlib import Path

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import PhaseGate, SwapGate, SXGate, SXdgGate, UGate

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPTS_ROOT.parent

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from simulate_ideal_vs_noisy_fidelity import (  # noqa: E402
    circuit_to_active_subcircuit,
    get_active_qubit_indices,
)

DEFAULT_MAX_QUBITS = 12


def load_circuit(path: str | Path) -> QuantumCircuit:
    """Load a circuit from a .qasm file (IBM / translated OpenQASM 2)."""
    custom = [
        qasm2.CustomInstruction("p", 1, 1, PhaseGate, builtin=True),
        qasm2.CustomInstruction("sx", 0, 1, SXGate, builtin=True),
        qasm2.CustomInstruction("sxdg", 0, 1, SXdgGate, builtin=True),
        qasm2.CustomInstruction("u", 3, 1, UGate, builtin=True),
        qasm2.CustomInstruction("swap", 0, 2, SwapGate, builtin=True),
    ]
    return qasm2.load(str(path), custom_instructions=custom)


def prepare_circuit(circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[int], int]:
    """Drop unused qubits and remap active lines to 0..n-1."""
    active_indices = get_active_qubit_indices(circuit)
    if not active_indices:
        raise ValueError("circuit has no gate operations on any qubit")

    n_register = circuit.num_qubits
    if len(active_indices) < n_register:
        circuit = circuit_to_active_subcircuit(circuit, active_indices)

    return circuit, active_indices, n_register


def find_qsyn_binary() -> Path:
    """Return path to the qsyn executable (build tree or PATH)."""
    candidates = [
        REPO_ROOT / "build" / "qsyn",
        REPO_ROOT / "qsyn",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    import shutil

    resolved = shutil.which("qsyn")
    if resolved is not None:
        return Path(resolved)
    raise FileNotFoundError(
        "qsyn executable not found. Build the project or add qsyn to PATH."
    )
