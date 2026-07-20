#!/usr/bin/env python3
"""Plot uniform vs log-PF cost comparisons from simulation metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

BENCHMARKS = [
    {
        "name": "Electron-4",
        "uniform": REPO / "experiments/electron-4",
        "log_pf": REPO / "experiments/electron-4-log-proxy-fidelity",
    },
    {
        "name": "Fermi-Hubbard-4",
        "uniform": REPO / "experiments/fermi-hubbard-4",
        "log_pf": REPO / "experiments/fermi-hubbard-4-log-proxy-fidelity",
    },
]


def load_metric(root: Path, field: str) -> dict[tuple[float, int], float]:
    by: dict[tuple[float, int], float] = {}
    for p in root.glob("t*-n*/metrics.json"):
        m = json.loads(p.read_text())
        if m.get(field) is None:
            continue
        by[(float(m["time"]), int(m["trotter_steps"]))] = float(m[field])
    return by


def plot_noise_relative_with_proxy(
    name: str,
    uniform: Path,
    log_pf: Path,
    out: Path,
) -> None:
    base_noise = load_metric(uniform, "noise_fidelity")
    pf_noise = load_metric(log_pf, "noise_fidelity")
    base_proxy = load_metric(uniform, "proxy_fidelity")
    pf_proxy = load_metric(log_pf, "proxy_fidelity")

    noise_keys = sorted(set(base_noise) & set(pf_noise))
    proxy_keys = sorted(set(base_proxy) & set(pf_proxy))
    noise_rel = {k: (pf_noise[k] - base_noise[k]) / base_noise[k] for k in noise_keys}
    proxy_rel = {k: (pf_proxy[k] - base_proxy[k]) / base_proxy[k] for k in proxy_keys}

    by_n_noise = defaultdict(list)
    by_n_proxy = defaultdict(list)
    for (_, n), r in noise_rel.items():
        by_n_noise[n].append(r)
    for (_, n), r in proxy_rel.items():
        by_n_proxy[n].append(r)

    ns = sorted(set(by_n_noise) & set(by_n_proxy))
    mean_noise = [np.mean(by_n_noise[n]) * 100 for n in ns]
    mean_proxy = [np.mean(by_n_proxy[n]) * 100 for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    times = sorted({t for t, _ in noise_keys})
    for i, t in enumerate(times):
        ns_t = sorted(n for (tt, n) in noise_keys if tt == t)
        vals = [noise_rel[(t, n)] * 100 for n in ns_t]
        ax.plot(
            ns_t,
            vals,
            marker="o",
            linewidth=2,
            markersize=6,
            color=cmap(i / max(len(times) - 1, 1)),
            label=f"t = {t:g}",
        )

    ax.plot(
        ns,
        mean_noise,
        color="black",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        label="mean noise (over t)",
    )
    ax.plot(
        ns,
        mean_proxy,
        color="#c44e52",
        linewidth=2.5,
        linestyle="-.",
        marker="D",
        markersize=7,
        label="mean proxy (over t)",
    )
    ax.axhline(0, color="0.5", linewidth=1)
    ax.set_xlabel("Trotter steps (n)")
    ax.set_ylabel("Relative improvement (%)")
    ax.set_xticks(ns)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=8, loc="upper left")

    ax.set_title(
        f"{name}: relative noise and proxy fidelity improvement\n"
        + r"$(F_{\mathrm{PF}}-F_{\mathrm{u}})/F_{\mathrm{u}}$"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_cost_comparison(
    name: str,
    field: str,
    ylabel: str,
    out_stem: str,
    uniform: Path,
    log_pf: Path,
    out: Path,
    *,
    skip_relative: bool = False,
) -> None:
    base = load_metric(uniform, field)
    pf = load_metric(log_pf, field)
    keys = sorted(set(base) & set(pf))
    rel = {k: (pf[k] - base[k]) / base[k] for k in keys}

    by_n_rel = defaultdict(list)
    by_n_base = defaultdict(list)
    by_n_pf = defaultdict(list)
    for (t, n), r in rel.items():
        by_n_rel[n].append(r)
        by_n_base[n].append(base[(t, n)])
        by_n_pf[n].append(pf[(t, n)])
    ns = sorted(by_n_rel)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, [np.mean(by_n_base[n]) for n in ns], marker="o", linewidth=2, label="uniform cost")
    ax.plot(
        ns,
        [np.mean(by_n_pf[n]) for n in ns],
        marker="o",
        linewidth=2,
        label="log_proxy_fidelity (exhaustive)",
    )
    ax.set_xlabel("Trotter steps (n)")
    ax.set_ylabel(f"Mean {ylabel} (avg over t)")
    ax.set_title(f"{name}: mean {ylabel} by cost function")
    ax.set_xticks(ns)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path = out / f"{out_stem}-cost-comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")

    if skip_relative:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    times = sorted({t for t, _ in keys})
    for i, t in enumerate(times):
        ns_t = sorted(n for (tt, n) in keys if tt == t)
        vals = [rel[(t, n)] * 100 for n in ns_t]
        ax.plot(
            ns_t,
            vals,
            marker="o",
            linewidth=2,
            markersize=6,
            color=cmap(i / max(len(times) - 1, 1)),
            label=f"t = {t:g}",
        )
    mean_by_n = [np.mean(by_n_rel[n]) * 100 for n in ns]
    ax.plot(ns, mean_by_n, color="black", linewidth=2.5, linestyle="--", marker="s", label="mean over t")
    ax.axhline(0, color="0.5", linewidth=1)
    ax.set_xlabel("Trotter steps (n)")
    ax.set_ylabel("Relative improvement (%)")
    ax.set_title(
        f"{name}: relative {ylabel} improvement\n"
        + r"$(F_{\mathrm{PF}}-F_{\mathrm{u}})/F_{\mathrm{u}}$"
    )
    ax.set_xticks(ns)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = out / f"{out_stem}-relative-improvement.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=["electron-4", "fermi-hubbard-4", "all"],
        default="all",
    )
    args = parser.parse_args()

    for bench in BENCHMARKS:
        slug = bench["name"].lower().replace(" ", "-")
        if args.benchmark != "all" and args.benchmark != slug:
            continue
        out = bench["log_pf"]
        plot_noise_relative_with_proxy(
            bench["name"],
            bench["uniform"],
            bench["log_pf"],
            out / "noise-fidelity-relative-improvement.png",
        )
        for field, ylabel, stem in [
            ("noise_fidelity", "noise fidelity", "noise-fidelity"),
            ("proxy_fidelity", "proxy fidelity", "proxy-fidelity"),
        ]:
            plot_cost_comparison(
                bench["name"],
                field,
                ylabel,
                stem,
                bench["uniform"],
                bench["log_pf"],
                out,
                skip_relative=(field == "noise_fidelity"),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
