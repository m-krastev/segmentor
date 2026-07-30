#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

export NAVIGATOR_DATA_DIR="${NAVIGATOR_DATA_DIR:-data/bomopi_resampled2_development15-v1}"
export NAVIGATOR_BOMOPI_MANIFEST="${NAVIGATOR_BOMOPI_MANIFEST:-experiments/navigator-bomopi-v1/development15.txt}"
export NAVIGATOR_TRAIN_CASE_IDS_FILE="${NAVIGATOR_TRAIN_CASE_IDS_FILE:-experiments/navigator-bomopi-v1/development_train12.txt}"
export NAVIGATOR_VAL_CASE_IDS_FILE="${NAVIGATOR_VAL_CASE_IDS_FILE:-experiments/navigator-bomopi-v1/development_validation3.txt}"
export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-filters-development64k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-filters-development64k-v1-validation}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-65536}"
export NAVIGATOR_EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-160}"
export NAVIGATOR_SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-160}"

exec "$PROJECT_ROOT/scripts/run_navigator_bomopi_filters_compact_cov500_gdt1_256k.sh" \
  "$@"
