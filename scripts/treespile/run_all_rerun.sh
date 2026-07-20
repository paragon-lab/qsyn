#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python3}"
LOGDIR="$REPO/experiments/logs"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d-%H%M%S)"

run_one() {
  local name="$1"
  local script="$2"
  local log="$LOGDIR/${name}-${TS}.log"
  echo "===== START $name $(date -Is) =====" | tee -a "$log"
  echo "log: $log"
  if "$PY" "$script" 2>&1 | tee -a "$log"; then
    echo "===== OK $name $(date -Is) =====" | tee -a "$log"
  else
    local rc=${PIPESTATUS[0]}
    echo "===== FAIL $name exit=$rc $(date -Is) =====" | tee -a "$log"
    return "$rc"
  fi
}

echo "qsyn: $(ls -l "$REPO/build/qsyn")"
echo "Starting full rerun at $(date -Is)"

# Simulation grids (Aer) — sequential to avoid memory contention
run_one electron-4-uniform \
  "$REPO/scripts/treespile/run_electron_4_grid.py"
run_one electron-4-log-proxy \
  "$REPO/scripts/treespile/run_electron_4_log_proxy_grid.py"
run_one fermi-hubbard-4-uniform \
  "$REPO/scripts/treespile/run_fermi_hubbard_grid.py"
run_one fermi-hubbard-4-log-proxy \
  "$REPO/scripts/treespile/run_fermi_hubbard_4_log_proxy_grid.py"

echo "===== ALL SIM GRIDS DONE $(date -Is) ====="
