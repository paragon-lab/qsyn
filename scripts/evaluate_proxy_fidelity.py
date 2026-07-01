## Compares proxy fidelity with final circuit fidelity

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import nativize_and_optimize_qasm as opt

CSV_NAME = "proxy_evaluation_fidelity.csv"
DEFAULT_SAMPLES = 500
DEFAULT_BACKEND = "fake_fez"
DEFAULT_BENCHMARK = "benchmark/fham/fermi-hubbard-18.fham"


def load_fake_backend(name: str):
    import qiskit_ibm_runtime.fake_provider as fake_provider

    parts = "".join(part.capitalize() for part in name.removeprefix("fake_").split("_"))
    for suffix in ("V2", ""):
        try:
            return getattr(fake_provider, f"Fake{parts}{suffix}")()
        except AttributeError:
            continue
    raise ValueError(f"Unknown fake backend: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare proxy infidelity with post-Qiskit circuit infidelity."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for proxy evaluation artifacts (CSV and sample QASM files)",
    )
    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of random trees to sample (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"Qiskit fake backend name (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help=f"Fermionic Hamiltonian benchmark path (default: {DEFAULT_BENCHMARK})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / CSV_NAME
    if csv_file.exists():
        csv_file.unlink()

    with csv_file.open("w") as f:
        f.write("TreeIndex,TotalProxyCost\n")

    cli_commands = (
        f"device fetch -f {args.backend}; "
        f"fham read {args.benchmark}; "
        f"fham eval-proxy -d {output_dir} -s {args.samples} -o {CSV_NAME}"
    )

    print(f"Compiling {args.samples} circuits via C++ engine...")
    process = subprocess.Popen(
        [str(REPO_ROOT / "qsyn"), "-c", cli_commands],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=REPO_ROOT,
    )

    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    process.wait()
    if process.returncode != 0:
        raise SystemExit(process.returncode)

    print("\nC++ Compilation complete. Optimizing and evaluating via Qiskit...")

    backend = load_fake_backend(args.backend)
    target = backend.target

    actual_fidelities = []
    valid_indices = []

    try:
        df_proxy = pd.read_csv(csv_file)

        for _, row in df_proxy.iterrows():
            tree_idx = int(row["TreeIndex"])
            qasm_filename = output_dir / f"sample_{tree_idx}.qasm"

            if not qasm_filename.exists():
                continue

            try:
                qc = QuantumCircuit.from_qasm_file(str(qasm_filename))

                qc_native = opt.nativize_gate_set(qc, backend)
                qc_opt = opt.post_mapping_optimize_preserve_connectivity(qc_native, backend)

                total_log_infidelity = 0.0
                for instruction in qc_opt.data:
                    gate_name = instruction.operation.name

                    if gate_name in ["barrier", "measure", "delay"]:
                        continue

                    qubits = tuple(qc_opt.find_bit(q).index for q in instruction.qubits)

                    try:
                        error = target[gate_name][qubits].error
                        if error is not None and error < 1.0:
                            total_log_infidelity += -np.log(1.0 - error)
                    except KeyError:
                        pass

                actual_fidelities.append(total_log_infidelity)
                valid_indices.append(tree_idx)

            except Exception as e:
                print(f"Failed to process {qasm_filename}: {e}")

        print("Evaluation complete. Graphing data...")

        df_filtered = df_proxy[df_proxy["TreeIndex"].isin(valid_indices)].copy()
        df_filtered["TotalActualCost"] = actual_fidelities
        df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna()

        plt.figure(figsize=(10, 6))
        plt.scatter(
            df_filtered["TotalProxyCost"],
            df_filtered["TotalActualCost"],
            alpha=0.6,
            color="purple",
            s=50,
            edgecolors="black",
        )

        plt.title(
            f"Circuit Fidelity ({args.benchmark}, {args.backend})",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Total Predicted Circuit Cost (Infidelity Proxy)", fontsize=12)
        plt.ylabel("Actual Cost (Post-Qiskit Infidelity)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        if df_filtered["TotalProxyCost"].nunique() > 1:
            z = np.polyfit(df_filtered["TotalProxyCost"], df_filtered["TotalActualCost"], 1)
            p = np.poly1d(z)

            correlation_matrix = np.corrcoef(
                df_filtered["TotalProxyCost"], df_filtered["TotalActualCost"]
            )
            r_val = correlation_matrix[0, 1]
            r_squared = r_val**2

            plt.plot(
                df_filtered["TotalProxyCost"],
                p(df_filtered["TotalProxyCost"]),
                color="red",
                linestyle="-",
                linewidth=2,
                label=(
                    f"Trendline (y = {z[0]:.2f}x + {z[1]:.2f})\n"
                    f"$R^2$={r_squared:.4f}"
                ),
            )
            plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nFAILED TO GRAPH DATA: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
