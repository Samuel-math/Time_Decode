#!/usr/bin/env bash
# Full-pipeline reproduction candidate. Historical metric reproduction is pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HORIZONS="${HORIZONS:-96 192 336 720}"
HORIZONS="${HORIZONS//,/ }"
for horizon in ${HORIZONS}; do
  case "$horizon" in
    96|192|336|720) ;;
    *) echo "Unsupported horizon: $horizon" >&2; exit 2 ;;
  esac
done
for horizon in ${HORIZONS}; do
  bash "$SCRIPT_DIR/best_configs/ettm1/h${horizon}.sh"
done
