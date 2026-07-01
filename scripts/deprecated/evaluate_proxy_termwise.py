import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_NAME = "proxy_evaluation.csv"
DEFAULT_SAMPLES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deprecated termwise proxy evaluation.")
    parser.add_argument("output_dir", type=Path, help="Directory for proxy evaluation artifacts")
    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of random trees to sample (default: {DEFAULT_SAMPLES})",
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
        f.write("TermProxyCost,IsolatedCNOTs,CNOTDiff\n")

    cli_commands = (
        "device fetch -f fake_oslo; "
        "fham read benchmark/fham/electron-4.fham; "
        f"fham eval-proxy -d {output_dir} -s {args.samples} -o {CSV_NAME}"
    )

    print("Compiling circuits and extracting term data in C++...")

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
        if "Evaluation saved" in line:
            process.terminate()
            break

    process.wait()

    print("\nEvaluation complete. Graphing...")

    try:
        df = pd.read_csv(csv_file)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        ax1.scatter(df["TermProxyCost"], df["IsolatedCNOTs"], alpha=0.05, color="#3498db", s=40)
        if df["TermProxyCost"].nunique() > 1:
            z1 = np.polyfit(df["TermProxyCost"], df["IsolatedCNOTs"], 1)
            ax1.plot(df["TermProxyCost"], np.poly1d(z1)(df["TermProxyCost"]), color="#2980b9", lw=2)

        ax1.set_title("2-Qubit Gates from Each Term", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Term Proxy Cost (Hops)", fontsize=11)
        ax1.set_ylabel("Physical 2Q Gates Generated", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.7)

        ax2.scatter(df["TermProxyCost"], df["CNOTDiff"], alpha=0.05, color="#e74c3c", s=40)
        if df["TermProxyCost"].nunique() > 1:
            z2 = np.polyfit(df["TermProxyCost"], df["CNOTDiff"], 1)
            ax2.plot(df["TermProxyCost"], np.poly1d(z2)(df["TermProxyCost"]), color="#c0392b", lw=2)

        ax2.set_title("Termwise 2Q GateContribution", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Term Proxy Cost (Hops)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.7)

        plt.suptitle("Term-Level Contributions", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nFAILED TO GRAPH DATA: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
