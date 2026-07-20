#!/usr/bin/env python3
"""Plot experiment fidelities vs Trotter steps from metrics.json artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO / "experiments/fermi-hubbard-4"
DEFAULT_GLOB = "experiments/fermi-hubbard-4/t*-n*/metrics.json"
DIR_RE = re.compile(r"(t[\dp]+)-n(\d+)$")

METRIC_SPECS = {
    "trotterization": {
        "field": "trotterization_fidelity",
        "ylabel": "Trotterization fidelity",
        "title_suffix": "trotterization fidelity vs n",
        "output": "trotter-fidelity.png",
        "overlay_reliability": False,
        "ylim": (0.0, 1.1),
    },
    "noise": {
        "field": "noise_fidelity",
        "ylabel": "Noise fidelity",
        "title_suffix": "noise fidelity vs n",
        "output": "noise-fidelity.png",
        "overlay_reliability": True,
        "ylim": (0.0, 1.0),
    },
    "total": {
        "field": "total_fidelity",
        "ylabel": "Total fidelity",
        "title_suffix": "total fidelity vs n",
        "output": "total-fidelity.png",
        "overlay_reliability": False,
        "ylim": (0.0, 1.0),
    },
}


def parse_dir_name(path: Path) -> tuple[float, int] | None:
    match = DIR_RE.search(path.parent.name)
    if not match:
        return None
    t_token, n_token = match.groups()
    t = float(t_token.replace("t", "").replace("p", "."))
    return t, int(n_token)


def load_metrics(
    pattern: str,
) -> tuple[
    dict[float, dict[int, dict[str, float]]],
    dict[int, float],
    dict[int, float],
]:
    by_time: dict[float, dict[int, dict[str, float]]] = {}
    esp_values: dict[int, list[float]] = defaultdict(list)
    proxy_values: dict[int, list[float]] = defaultdict(list)

    for metrics_path in sorted(REPO.glob(pattern)):
        parsed = parse_dir_name(metrics_path)
        if parsed is None:
            continue
        t, n = parsed
        data = json.loads(metrics_path.read_text())
        by_time.setdefault(t, {})[n] = {
            key: float(data[key])
            for key in (
                "trotterization_fidelity",
                "noise_fidelity",
                "total_fidelity",
                "esp",
                "proxy_fidelity",
            )
            if data.get(key) is not None
        }
        if data.get("esp") is not None:
            esp_values[n].append(float(data["esp"]))
        if data.get("proxy_fidelity") is not None:
            proxy_values[n].append(float(data["proxy_fidelity"]))

    if not by_time:
        raise FileNotFoundError(f"No metrics found for pattern: {pattern}")

    esp_by_n = {n: mean(values) for n, values in sorted(esp_values.items())}
    proxy_by_n = {n: mean(values) for n, values in sorted(proxy_values.items())}
    return by_time, esp_by_n, proxy_by_n


def plot_fidelity(
    by_time: dict[float, dict[int, dict[str, float]]],
    *,
    field: str,
    ylabel: str,
    title: str,
    output: Path,
    overlay_reliability: bool = False,
    esp_by_n: dict[int, float] | None = None,
    proxy_by_n: dict[int, float] | None = None,
    ylim: tuple[float, float] = (0.0, 1.0),
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.get_cmap("viridis")
    times = sorted(by_time)
    for i, t in enumerate(times):
        ns = sorted(by_time[t])
        fidelities = [by_time[t][n][field] for n in ns]
        color = cmap(i / max(len(times) - 1, 1))
        ax.plot(
            ns,
            fidelities,
            marker="o",
            linewidth=2,
            markersize=6,
            color=color,
            label=f"t = {t:g}",
        )

    if overlay_reliability and esp_by_n:
        ns = sorted(esp_by_n)
        ax.plot(
            ns,
            [esp_by_n[n] for n in ns],
            marker="s",
            linewidth=2.5,
            markersize=6,
            color="black",
            linestyle="--",
            label="ESP",
            zorder=10,
        )

    if overlay_reliability and proxy_by_n:
        ns = sorted(proxy_by_n)
        ax.plot(
            ns,
            [proxy_by_n[n] for n in ns],
            marker="D",
            linewidth=2.5,
            markersize=6,
            color="0.35",
            linestyle=":",
            label="Proxy fidelity",
            zorder=10,
        )

    y_min, y_max = ylim
    ax.set_xlabel("Trotter steps (n)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min, y_max)
    tick_step = 0.1
    n_ticks = int(round((y_max - y_min) / tick_step)) + 1
    yticks = [y_min + i * tick_step for i in range(n_ticks)]
    ax.set_yticks(yticks)
    # Keep headroom above 1.0 without labeling the upper bound tick.
    ax.set_yticklabels(
        ["" if abs(y - y_max) < 1e-9 and y_max > 1.0 else f"{y:.1f}" for y in yticks]
    )
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.5, 5.5)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="Evolution time", ncol=2, fontsize=9)
    ax.set_title(title)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Wrote {output}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        default=DEFAULT_GLOB,
        help=f"glob for metrics.json files (default: {DEFAULT_GLOB})",
    )
    parser.add_argument(
        "--metric",
        choices=[*METRIC_SPECS, "all"],
        default="all",
        help="which fidelity plot to generate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR,
        help="directory for output images",
    )
    parser.add_argument(
        "--title-prefix",
        default="Fermi-Hubbard-4 treespile",
        help="prefix for plot titles (default: Fermi-Hubbard-4 treespile)",
    )
    parser.add_argument("--show", action="store_true", help="display interactively")
    args = parser.parse_args()

    by_time, esp_by_n, proxy_by_n = load_metrics(args.pattern)
    metrics = list(METRIC_SPECS) if args.metric == "all" else [args.metric]

    for metric in metrics:
        spec = METRIC_SPECS[metric]
        plot_fidelity(
            by_time,
            field=spec["field"],
            ylabel=spec["ylabel"],
            title=f"{args.title_prefix}: {spec['title_suffix']}",
            output=args.output_dir / spec["output"],
            overlay_reliability=spec["overlay_reliability"],
            esp_by_n=esp_by_n,
            proxy_by_n=proxy_by_n,
            ylim=spec["ylim"],
            show=args.show,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
