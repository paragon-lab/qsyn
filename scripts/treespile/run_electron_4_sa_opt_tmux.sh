#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
LOGDIR="$REPO/experiments/logs"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="$LOGDIR/electron-4-sa-opt-${TS}.log"
echo "===== START electron-4 SA opt grid $(date -Is) =====" | tee -a "$LOG"
echo "log: $LOG" | tee -a "$LOG"
echo "qsyn: $(ls -l "$REPO/build/qsyn")" | tee -a "$LOG"
set +e
python3 "$REPO/scripts/treespile/run_electron_4_sa_opt_grid.py" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
if [[ "$rc" -eq 0 ]]; then
  echo "===== OK $(date -Is) =====" | tee -a "$LOG"
else
  echo "===== FAIL exit=$rc $(date -Is) =====" | tee -a "$LOG"
fi
echo "===== ALL DONE $(date -Is) =====" | tee -a "$LOG"
exit "$rc"
