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
BATCH_SAMPLES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deprecated proxy vs CNOT count evaluation.")
    parser.add_argument("output_dir", type=Path, help="Directory for proxy evaluation artifacts")
    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of valid trees to collect (default: {DEFAULT_SAMPLES})",
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
        f.write("ProxyCost,TrueCNOTs\n")

    cli_commands = (
        "device fetch -f fake_oslo; "
        "fham read benchmark/fham/electron-4.fham; "
        f"fham eval-proxy -d {output_dir} -s {BATCH_SAMPLES} -o {CSV_NAME}"
    )

    current_samples = 0

    print(f"Goal: Collect {args.samples} valid quantum circuits.")

    while current_samples < args.samples:
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

        with csv_file.open("r") as f:
            lines = f.readlines()
        current_samples = max(0, len(lines) - 1)
        print(f"\n---> PROGRESS: {current_samples} / {args.samples} valid trees saved.\n")

    try:
        df = pd.read_csv(csv_file)
        df = df.dropna().head(args.samples)

        plt.figure(figsize=(10, 6))
        plt.scatter(
            df["ProxyCost"],
            df["TrueCNOTs"],
            alpha=0.5,
            color="#2c3e50",
            edgecolors="w",
            s=60,
        )
        plt.title(
            "Proxy Cost (number of two-qubit tree interactions) vs Actual Number of CNOTs",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Fast Tree Proxy Cost (Hops)", fontsize=12)
        plt.ylabel("True Physical CNOT Count", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        if df["ProxyCost"].nunique() > 1:
            z = np.polyfit(df["ProxyCost"], df["TrueCNOTs"], 1)
            p = np.poly1d(z)
            plt.plot(df["ProxyCost"], p(df["ProxyCost"]), color="#e74c3c", linestyle="-", linewidth=2, label="Linear Trend")
            plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nFAILED TO GRAPH DATA: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
