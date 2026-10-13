#!/usr/bin/env python3
"""Plot SA optimization-grid fidelities and runtimes for electron-4."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments/electron-4-sa-opt"

METHODS = [
    {
        "key": "pauli-weight",
        "label": "Pauli weight (o1)",
        "dir": REPO / "experiments/electron-4-sa-pauli-weight",
        "color": "#4c72b0",
        "marker": "o",
    },
    {
        "key": "uniform-apsp",
        "label": "Uniform APSP (o2)",
        "dir": REPO / "experiments/electron-4-sa-uniform-apsp",
        "color": "#55a868",
        "marker": "s",
    },
    {
        "key": "log-proxy-fidelity",
        "label": "Log proxy PF (o2)",
        "dir": REPO / "experiments/electron-4-sa-log-proxy-fidelity",
        "color": "#c44e52",
        "marker": "D",
    },
]

FIDELITY_SPECS = [
    ("trotterization_fidelity", "Trotterization fidelity", "trotterization-fidelity.png", (0.0, 1.1)),
    ("noise_fidelity", "Noise fidelity", "noise-fidelity.png", (0.0, 1.0)),
    ("total_fidelity", "Total fidelity", "total-fidelity.png", (0.0, 1.0)),
]

RUNTIME_FIELDS = [
    ("synthesis_runtime_sec", "Synthesis wall time (s)", "synthesis-runtime.png"),
    ("experiment_runtime_sec", "Experiment wall time (s)", "experiment-runtime.png"),
]


def load_method(root: Path) -> dict[tuple[float, int], dict[str, float]]:
    by: dict[tuple[float, int], dict[str, float]] = {}
    for p in sorted(root.glob("t*-n*/metrics.json")):
        m = json.loads(p.read_text())
        key = (float(m["time"]), int(m["trotter_steps"]))
        row: dict[str, float] = {}
        for k in (
            "trotterization_fidelity",
            "noise_fidelity",
            "total_fidelity",
            "esp",
            "proxy_fidelity",
            "synthesis_runtime_sec",
            "experiment_runtime_sec",
        ):
            if m.get(k) is not None:
                row[k] = float(m[k])
            std_key = f"{k}_std"
            if m.get(std_key) is not None:
                row[std_key] = float(m[std_key])
        by[key] = row
    return by


def mean_by_n(
    data: dict[tuple[float, int], dict[str, float]], field: str
) -> tuple[list[int], list[float]]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for (_, n), row in data.items():
        if field in row:
            buckets[n].append(row[field])
    ns = sorted(buckets)
    return ns, [float(np.mean(buckets[n])) for n in ns]


def plot_fidelity_comparison(
    loaded: list[tuple[dict, dict[tuple[float, int], dict[str, float]]]],
    field: str,
    ylabel: str,
    out: Path,
    ylim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    all_ns: set[int] = set()
    for method, data in loaded:
        ns, means = mean_by_n(data, field)
        if not ns:
            continue
        all_ns.update(ns)
        ax.plot(
            ns,
            means,
            marker=method["marker"],
            linewidth=2.5,
            markersize=7,
            color=method["color"],
            label=method["label"],
        )

    y_min, y_max = ylim
    ax.set_xlabel("Trotter steps (n)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min, y_max)
    ns_sorted = sorted(all_ns)
    ax.set_xticks(ns_sorted)
    if ns_sorted:
        ax.set_xlim(ns_sorted[0] - 0.5, ns_sorted[-1] + 0.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    ax.set_title(f"Electron-4 SA optimization: mean {ylabel.lower()} vs n")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_runtime_comparison(
    loaded: list[tuple[dict, dict[tuple[float, int], dict[str, float]]]],
    field: str,
    ylabel: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    all_ns: set[int] = set()
    for method, data in loaded:
        ns, means = mean_by_n(data, field)
        if not ns:
            continue
        all_ns.update(ns)
        ax.plot(
            ns,
            means,
            marker=method["marker"],
            linewidth=2.5,
            markersize=7,
            color=method["color"],
            label=method["label"],
        )

    ns_sorted = sorted(all_ns)
    ax.set_xlabel("Trotter steps (n)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ns_sorted)
    if ns_sorted:
        ax.set_xlim(ns_sorted[0] - 0.5, ns_sorted[-1] + 0.5)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    ax.set_title(f"Electron-4 SA optimization: mean {ylabel.lower()} vs n")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_per_method_fidelity(
    method: dict,
    data: dict[tuple[float, int], dict[str, float]],
    field: str,
    ylabel: str,
    out: Path,
    ylim: tuple[float, float],
) -> None:
    by_time: dict[float, dict[int, tuple[float, float | None]]] = defaultdict(dict)
    for (t, n), row in data.items():
        if field in row:
            by_time[t][n] = (row[field], row.get(f"{field}_std"))
    if not by_time:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    times = sorted(by_time)
    for i, t in enumerate(times):
        ns = sorted(by_time[t])
        ys = [by_time[t][n][0] for n in ns]
        yerr = [by_time[t][n][1] or 0.0 for n in ns]
        color = cmap(i / max(len(times) - 1, 1))
        if any(e > 0 for e in yerr):
            ax.errorbar(
                ns,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=3,
                color=color,
                label=f"t = {t:g}",
            )
        else:
            ax.plot(
                ns,
                ys,
                marker="o",
                linewidth=2,
                markersize=6,
                color=color,
                label=f"t = {t:g}",
            )

    y_min, y_max = ylim
    all_ns = sorted({n for series in by_time.values() for n in series})
    ax.set_xlabel("Trotter steps (n)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(all_ns)
    if all_ns:
        ax.set_xlim(all_ns[0] - 0.5, all_ns[-1] + 0.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Evolution time", ncol=2, fontsize=8)
    title = f"Electron-4 {method['label']}: {ylabel.lower()} vs n"
    if any(row.get(f"{field}_std", 0) for row in data.values()):
        title += " (mean ± std over 3 SA runs)"
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help=f"comparison plots directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--min-metrics",
        type=int,
        default=1,
        help="require at least this many metrics.json per method",
    )
    args = parser.parse_args()

    loaded: list[tuple[dict, dict[tuple[float, int], dict[str, float]]]] = []
    for method in METHODS:
        data = load_method(method["dir"])
        print(f"{method['key']}: {len(data)} metrics in {method['dir']}")
        if len(data) < args.min_metrics:
            raise SystemExit(
                f"Need >= {args.min_metrics} metrics for {method['key']}, found {len(data)}"
            )
        loaded.append((method, data))

        for field, ylabel, fname, ylim in FIDELITY_SPECS:
            plot_per_method_fidelity(
                method,
                data,
                field,
                ylabel,
                method["dir"] / fname,
                ylim,
            )

    for field, ylabel, fname, ylim in FIDELITY_SPECS:
        plot_fidelity_comparison(loaded, field, ylabel, args.output_dir / fname, ylim)

    for field, ylabel, fname in RUNTIME_FIELDS:
        plot_runtime_comparison(loaded, field, ylabel, args.output_dir / fname)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
