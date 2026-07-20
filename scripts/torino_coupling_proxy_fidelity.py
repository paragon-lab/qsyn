#!/usr/bin/env python3
"""
Adjusted per-coupling proxy fidelity on FakeTorino (IBM Torino).

    PF(g) = sqrt(
        (1 - gate_error)^2
        * product over {q1, q2} of (2/3 exp(-t/T2) + 1/3 exp(-t/T1))
    )

Thermal relaxation is approximated as a depolarizing channel via the same
multiplicative factor used in the circuit proxy fidelity (Wu et al. 2026).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeTorino


def thermal_factor(gate_time_s: float, t1_s: float, t2_s: float) -> float:
    return (2.0 / 3.0) * np.exp(-gate_time_s / t2_s) + (1.0 / 3.0) * np.exp(
        -gate_time_s / t1_s
    )


def coupling_proxy_fidelity(
    gate_error: float,
    gate_time_s: float,
    t1_q1: float,
    t2_q1: float,
    t1_q2: float,
    t2_q2: float,
) -> float:
    return float(
        np.sqrt(
            (1.0 - gate_error) ** 2
            * thermal_factor(gate_time_s, t1_q1, t2_q1)
            * thermal_factor(gate_time_s, t1_q2, t2_q2)
        )
    )


def collect_rows(backend: FakeTorino) -> list[tuple[int, int, float]]:
    props = backend.properties()
    cz = backend.target["cz"]
    rows: list[tuple[int, int, float]] = []

    for (q1, q2), inst in cz.items():
        if q1 > q2:
            continue  # undirected; both directions share identical calibration
        err = inst.error
        t = inst.duration
        if err is None or t is None:
            continue
        t1_1, t2_1 = props.t1(q1), props.t2(q1)
        t1_2, t2_2 = props.t1(q2), props.t2(q2)
        if None in (t1_1, t2_1, t1_2, t2_2) or min(t1_1, t2_1, t1_2, t2_2) <= 0:
            continue
        pf = coupling_proxy_fidelity(err, t, t1_1, t2_1, t1_2, t2_2)
        rows.append((q1, q2, float(pf)))

    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("experiments/torino-coupling-pf"),
        help="Directory for CSV and histogram (default: experiments/torino-coupling-pf)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "coupling_proxy_fidelity.csv"
    plot_path = out_dir / "coupling_proxy_fidelity_hist.png"

    rows = collect_rows(FakeTorino())
    pfs = np.array([pf for _, _, pf in rows], dtype=float)
    pfs_stats = pfs[pfs > 0]
    mean_pf = float(np.mean(pfs_stats))
    std_pf = float(np.std(pfs_stats))

    with csv_path.open("w") as f:
        f.write("q1,q2,proxy_fidelity\n")
        for q1, q2, pf in rows:
            f.write(f"{q1},{q2},{pf:.12g}\n")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(pfs_stats, bins=50, color="#2c6eaf", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Gate Proxy Fidelity")
    ax.set_ylabel("Count")
    ax.set_title(
        f"IBM Torino (FakeTorino) CZ coupling PF\n"
        f"mean = {mean_pf:.6f}, std = {std_pf:.6f} "
        f"(n = {len(pfs_stats)}, excluding {len(pfs) - len(pfs_stats)} zeros)"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"couplings: {len(rows)}")
    print(f"mean PF:   {mean_pf:.6f}")
    print(f"std PF:    {std_pf:.6f}")
    print(f"CSV:       {csv_path}")
    print(f"histogram: {plot_path}")


if __name__ == "__main__":
    main()
