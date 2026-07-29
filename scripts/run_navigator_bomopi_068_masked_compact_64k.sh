#!/usr/bin/env bash
set -euo pipefail

# Controlled follow-up to the dense 2,196-action masked screen. This changes
# only categorical support: 26 lattice directions at each of six step lengths.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-068-masked-compact-64k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-068-masked-compact-64k-v1-validation}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-65536}"
export NAVIGATOR_EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-160}"
export NAVIGATOR_SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-160}"
export NAVIGATOR_CATEGORICAL_ACTION_SUPPORT=direction_length

exec "$SCRIPT_DIR/run_navigator_bomopi_068_masked_64k.sh" "$@"
