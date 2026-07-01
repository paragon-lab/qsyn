## Evaluates termwise contribution to overall circuit depth, as well as termwise proxy depth
## TO USE: Change fham_cmd.cpp eval func to use evaluate_proxy_termwise_depth

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_NAME = "proxy_evaluation.csv"
DEFAULT_SAMPLES = 50
DEFAULT_BACKEND = "fake_torino"
DEFAULT_BENCHMARK = "benchmark/fham/electron-4.fham"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deprecated termwise depth proxy evaluation.")
    parser.add_argument("output_dir", type=Path, help="Directory for proxy evaluation artifacts")
    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of random trees to sample (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument("-b", "--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / CSV_NAME
    if csv_file.exists():
        csv_file.unlink()

    with csv_file.open("w") as f:
        f.write("TermIndex,TermProxyCost,CriticalDepth\n")

    cli_commands = (
        f"device fetch -f {args.backend}; "
        f"fham read {args.benchmark}; "
        f"fham eval-proxy -d {output_dir} -s {args.samples} -o {CSV_NAME}"
    )

    print(f"Compiling {args.samples} trees for {args.benchmark} on {args.backend}...")

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

    print("\nEvaluation complete. Graphing data...")

    try:
        df = pd.read_csv(csv_file)
        df = df[df["CriticalDepth"] > 0]

        if len(df) == 0:
            print("Error: No terms contributed to overall circuit depth.")
            raise SystemExit(1)

        plt.figure(figsize=(10, 6))
        plt.scatter(df["TermProxyCost"], df["CriticalDepth"], alpha=0.1, color="blue", s=40)

        plt.title(
            f"Termwise Depth Contribution ({args.benchmark}, {args.backend})",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Individual Term Proxy Cost (Logical Hops)", fontsize=12)
        plt.ylabel("Depth Added to Total Circuit Depth (# Gates)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        if df["TermProxyCost"].nunique() > 1:
            z = np.polyfit(df["TermProxyCost"], df["CriticalDepth"], 1)
            np.poly1d(z)
            plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nFAILED TO GRAPH DATA: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
