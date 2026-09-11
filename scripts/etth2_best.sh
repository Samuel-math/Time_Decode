#!/usr/bin/env bash
# Historical best configuration index; scratch reproduction is not verified.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${TD_STAGE:-}" != finetune ]; then
  echo "Historical best from-scratch provenance is incomplete for etth2." >&2
  echo "For checkpoint-based fine-tuning only: TD_STAGE=finetune HORIZONS=96 bash scripts/etth2_best.sh" >&2
  exit 2
fi
HORIZONS="${HORIZONS:-96 192 336 720}"
HORIZONS="${HORIZONS//,/ }"
for horizon in $HORIZONS; do
  case "$horizon" in 96|192|336|720) ;; *) echo "Invalid horizon: $horizon" >&2; exit 2;; esac
  test -f "$SCRIPT_DIR/best_configs/etth2/h$horizon.sh" || { echo "Missing verified parameters: etth2-$horizon" >&2; exit 2; }
done
for horizon in $HORIZONS; do
  bash "$SCRIPT_DIR/best_configs/etth2/h$horizon.sh"
done
