#!/usr/bin/env python3
"""
Translate a circuit to a backend's gate set and run connectivity-preserving
optimizations (no layout or routing). Output is OpenQASM 2 with the same
qubit indices as the input.

Use this before simulate_ideal_vs_noisy_fidelity.py when you want to
optimize the circuit for a specific backend while preserving which
physical qubits each gate acts on.

Usage:
  python translate_and_optimize_qasm.py circuit.qasm --backend fake_algiers -o circuit.opt.qasm
  python translate_and_optimize_qasm.py circuit.qasm --backend fake_algiers  # writes circuit.opt.qasm by default
"""

import argparse
import json
import sys
from pathlib import Path

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as _sel
from qiskit.circuit.library import PhaseGate, RZGate, SXGate, SXdgGate
from qiskit.transpiler import PassManager, TranspilerError
from qiskit.transpiler.passes import (
    BasisTranslator,
    Collect2qBlocks,
    CommutativeCancellation,
    ConsolidateBlocks,
    Optimize1qGatesDecomposition,
    UnitarySynthesis,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "get_backend.py").exists():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from get_backend import get_fake_backend, get_real_backend
        HAS_GET_BACKEND = True
    except ImportError:
        HAS_GET_BACKEND = False
else:
    HAS_GET_BACKEND = False

try:
    import qiskit_ibm_runtime.fake_provider as fake_provider
    HAS_FAKE_PROVIDER = True
except ImportError:
    HAS_FAKE_PROVIDER = False


def load_circuit(path: str) -> QuantumCircuit:
    """Load a circuit from a .qasm file."""
    custom = [
        qasm2.CustomInstruction("p", 1, 1, PhaseGate, builtin=True),
        qasm2.CustomInstruction("sx", 0, 1, SXGate, builtin=True),
        qasm2.CustomInstruction("sxdg", 0, 1, SXdgGate, builtin=True),
    ]
    return qasm2.load(path, custom_instructions=custom)


def _get_backend_basis(backend):
    """Return (basis_gates_list, target or None) for the backend."""
    target = getattr(backend, "target", None)
    if target is not None:
        basis = list(target.operation_names)
        return basis, target
    cfg = getattr(backend, "configuration", lambda: None)()
    basis = list(getattr(cfg, "basis_gates", []) or [])
    return basis, None


def _normalize_basis_gates(basis_gates: list[str]) -> list[str]:
    return [g.lower() for g in basis_gates]


def validate_basis_gates(basis_gates: list[str]) -> None:
    """Reject gate sets Qiskit cannot use to translate typical 1q/2q unitaries."""
    if not basis_gates:
        raise ValueError("basis gate set is empty")

    probe = QuantumCircuit(2)
    probe.h(0)
    probe.cx(0, 1)
    probe.append(RZGate(0.1), [0])
    probe.append(SXGate(), [1])

    try:
        PassManager([BasisTranslator(_sel, basis_gates, None)]).run(probe)
    except TranspilerError as exc:
        raise ValueError(
            f"gate set [{', '.join(basis_gates)}] is not accepted by Qiskit "
            f"(cannot translate standard unitaries): {exc}"
        ) from exc


def load_basis_gates_file(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not all(isinstance(g, str) for g in data):
        raise ValueError(f"{path}: expected a JSON array of gate name strings")
    return _normalize_basis_gates(data)


def translate_gate_set_basis(circuit: QuantumCircuit, basis_gates: list[str]) -> QuantumCircuit:
    """Convert the circuit to the given basis (no IBM backend object)."""
    if not basis_gates:
        return circuit
    pm = PassManager([BasisTranslator(_sel, basis_gates, None)])
    return pm.run(circuit)


def post_mapping_optimize_basis(circuit: QuantumCircuit, basis_gates: list[str]) -> QuantumCircuit:
    """Connectivity-preserving optimizations for a fixed basis gate list."""
    passes = []
    if basis_gates:
        passes.append(Optimize1qGatesDecomposition(basis=basis_gates))
    passes.append(CommutativeCancellation())
    passes.append(Collect2qBlocks())
    if basis_gates:
        passes.append(ConsolidateBlocks(basis_gates=basis_gates))
        passes.append(UnitarySynthesis(basis_gates=basis_gates, target=None))
        # Resynthesize any 1q runs left after 2q consolidation, then force basis.
        passes.append(Optimize1qGatesDecomposition(basis=basis_gates))
        passes.append(BasisTranslator(_sel, basis_gates, None))
    return PassManager(passes).run(circuit)


def translate_gate_set(circuit: QuantumCircuit, backend) -> QuantumCircuit:
    """Convert the circuit to the backend's native gate set only (no layout/routing)."""
    basis_gates, target = _get_backend_basis(backend)
    if not basis_gates:
        return circuit
    pm = PassManager([BasisTranslator(_sel, basis_gates, target)])
    return pm.run(circuit)


def post_mapping_optimize_preserve_connectivity(circuit: QuantumCircuit, backend) -> QuantumCircuit:
    """
    Run optimizations that do not change qubit layout or add SWAPs: merge 1q gates,
    cancel commuting gates, consolidate 2q blocks. Preserves which qubits each gate acts on.
    UnitarySynthesis is run last so that any consolidated unitaries are synthesized into the
    backend basis, keeping the final circuit hardware-compliant.
    """
    basis_gates, target = _get_backend_basis(backend)
    passes = []
    if basis_gates:
        passes.append(Optimize1qGatesDecomposition(basis=basis_gates, target=target))
    else:
        passes.append(Optimize1qGatesDecomposition())
    passes.append(CommutativeCancellation())
    passes.append(Collect2qBlocks())
    if basis_gates:
        passes.append(ConsolidateBlocks(basis_gates=basis_gates))
        # Synthesize consolidated 2q unitaries (and other non-basis gates) into backend basis.
        passes.append(UnitarySynthesis(basis_gates=basis_gates, target=target))
        passes.append(Optimize1qGatesDecomposition(basis=basis_gates, target=target))
        passes.append(BasisTranslator(_sel, basis_gates, target))
    return PassManager(passes).run(circuit)


def main():
    parser = argparse.ArgumentParser(
        description="Translate circuit to backend gate set and run connectivity-preserving optimizations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("circuit", type=str, help="Path to circuit file (.qasm)")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend name (e.g. fake_algiers). Mutually exclusive with --basis-gates-file.",
    )
    parser.add_argument(
        "--basis-gates-file",
        type=str,
        default=None,
        help="JSON file containing a list of basis gate names (from qsyn Device gate set).",
    )
    parser.add_argument(
        "--use-real-backend",
        action="store_true",
        help="Use real IBM Quantum backend (requires IBMQ_API_KEY).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path. Default: input stem with .opt.qasm suffix.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Only translate the gate set; do not run post-mapping optimizations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages (qsyn passes this when logger level is info or more verbose).",
    )
    args = parser.parse_args()

    circuit_path = Path(args.circuit)
    if not circuit_path.exists():
        print(f"Error: circuit file not found: {circuit_path}", file=sys.stderr)
        return 1

    if args.basis_gates_file and args.backend:
        print("Error: use only one of --backend and --basis-gates-file", file=sys.stderr)
        return 1
    if not args.basis_gates_file and not args.backend:
        print("Error: specify --backend or --basis-gates-file", file=sys.stderr)
        return 1

    try:
        circuit = load_circuit(str(circuit_path))
    except Exception as e:
        print(f"Error loading circuit: {e}", file=sys.stderr)
        return 1

    basis_gates: list[str] | None = None
    backend = None

    if args.basis_gates_file:
        try:
            basis_gates = load_basis_gates_file(args.basis_gates_file)
            validate_basis_gates(basis_gates)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        try:
            circuit = translate_gate_set_basis(circuit, basis_gates)
            if not args.no_optimize:
                circuit = post_mapping_optimize_basis(circuit, basis_gates)
                if args.verbose:
                    print("Applied post-mapping optimizations (connectivity preserved).")
        except TranspilerError as e:
            print(f"Error: Qiskit rejected the gate set for this circuit: {e}", file=sys.stderr)
            return 1
    else:
        if args.use_real_backend and HAS_GET_BACKEND:
            backend = get_real_backend(args.backend, verbose=True)
        else:
            if HAS_GET_BACKEND:
                backend = get_fake_backend(args.backend, verbose=True)
            elif HAS_FAKE_PROVIDER:
                parts = args.backend.replace("fake_", "").replace("_", " ").title().replace(" ", "")
                try:
                    backend = getattr(fake_provider, f"Fake{parts}V2")()
                except AttributeError:
                    backend = getattr(fake_provider, f"Fake{parts}")()
            else:
                print("Error: need qiskit_ibm_runtime (fake provider) or get_backend.py", file=sys.stderr)
                return 1

        if backend is None:
            print("Error: could not get backend", file=sys.stderr)
            return 1

        circuit = translate_gate_set(circuit, backend)
        if not args.no_optimize:
            circuit = post_mapping_optimize_preserve_connectivity(circuit, backend)
            if args.verbose:
                print("Applied post-mapping optimizations (connectivity preserved).")

    out_path = args.output
    if out_path is None:
        out_path = circuit_path.with_stem(circuit_path.stem + ".opt").with_suffix(".qasm")
    else:
        out_path = Path(out_path)

    try:
        with open(out_path, "w") as f:
            qasm2.dump(circuit, f)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
