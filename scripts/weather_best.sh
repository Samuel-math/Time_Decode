#!/usr/bin/env bash
# Full codebook -> token pretraining -> fine-tuning; reproduction pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HORIZONS="${HORIZONS:-96 192 336 720}"
HORIZONS="${HORIZONS//,/ }"
for horizon in $HORIZONS; do
  case "$horizon" in 96|192|336|720) ;; *) echo "Invalid horizon: $horizon" >&2; exit 2;; esac
  test -f "$SCRIPT_DIR/best_configs/weather/h$horizon.sh" || { echo "Missing verified parameters: weather-$horizon" >&2; exit 2; }
done
for horizon in $HORIZONS; do
  bash "$SCRIPT_DIR/best_configs/weather/from_scratch.sh" "$horizon"
done
