#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_SUFFIX="data/bomopi_resampled2_development15-v1"

export NAVIGATOR_RELOAD_CHECKPOINT_PATH="${NAVIGATOR_RELOAD_CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/navigator-bomopi-gru-filters-pretrained-development64k-v1/${DATA_SUFFIX}/checkpoint_65536.pth}"
export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-filters-pretrained-development256k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-filters-pretrained-development256k-v1-validation}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-256000}"
export NAVIGATOR_EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-1000}"
export NAVIGATOR_SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-500}"

exec "$SCRIPT_DIR/run_navigator_bomopi_filters_pretrained_development64k.sh" \
  "$@"
