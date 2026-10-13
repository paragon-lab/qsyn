#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
LOGDIR="$REPO/experiments/logs"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="$LOGDIR/electron-4-sa-pauli-weight-x3-${TS}.log"
echo "===== START electron-4 SA pauli-weight 3-replicate average $(date -Is) =====" | tee -a "$LOG"
echo "log: $LOG" | tee -a "$LOG"
echo "qsyn: $(ls -l "$REPO/build/qsyn")" | tee -a "$LOG"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set +e
python3 "$REPO/scripts/treespile/run_electron_4_sa_opt_grid.py" \
  --methods pauli-weight \
  --replicates 3 \
  --reuse-legacy \
  --force \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [[ "$rc" -eq 0 ]]; then
  echo "===== plotting $(date -Is) =====" | tee -a "$LOG"
  python3 "$REPO/scripts/treespile/plot_sa_opt_comparison.py" 2>&1 | tee -a "$LOG"
  python3 - <<'PY' 2>&1 | tee -a "$LOG"
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

root = Path("experiments/electron-4-sa-pauli-weight")
out = root / "synthesis-runtime.png"
by_time = defaultdict(dict)
by_n = defaultdict(list)
for p in sorted(root.glob("t*-n*/metrics.json")):
    m = json.loads(p.read_text())
    if m.get("synthesis_runtime_sec") is None:
        continue
    t, n = float(m["time"]), int(m["trotter_steps"])
    rt = float(m["synthesis_runtime_sec"])
    std = float(m.get("synthesis_runtime_sec_std") or 0.0)
    by_time[t][n] = (rt, std)
    by_n[n].append(rt)

fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.get_cmap("viridis")
times = sorted(by_time)
for i, t in enumerate(times):
    ns = sorted(by_time[t])
    ys = [by_time[t][n][0] for n in ns]
    yerr = [by_time[t][n][1] for n in ns]
    ax.errorbar(
        ns, ys, yerr=yerr, marker="o", linewidth=2, markersize=6, capsize=3,
        color=cmap(i / max(len(times) - 1, 1)), label=f"t = {t:g}",
    )
ns = sorted(by_n)
ax.plot(
    ns, [float(np.mean(by_n[n])) for n in ns], color="black", linewidth=2.5,
    linestyle="--", marker="s", label="mean (over t)", zorder=10,
)
ax.set_xlabel("Trotter steps (n)")
ax.set_ylabel("Synthesis wall time (s)")
ax.set_xticks(ns)
ax.set_xlim(ns[0] - 0.5, ns[-1] + 0.5)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(title="Evolution time", ncol=2, fontsize=8)
ax.set_title("Electron-4 SA Pauli weight: synthesis runtime vs n (mean ± std over 3 SA runs)")
fig.tight_layout()
fig.savefig(out, dpi=160)
print(f"Wrote {out}")
PY
fi
set -e
if [[ "$rc" -eq 0 ]]; then
  echo "===== OK $(date -Is) =====" | tee -a "$LOG"
else
  echo "===== FAIL exit=$rc $(date -Is) =====" | tee -a "$LOG"
fi
echo "===== ALL DONE $(date -Is) =====" | tee -a "$LOG"
exit "$rc"
