#!/usr/bin/env bash
set -euo pipefail

# Boundary-safe follow-up to the repaired-068 screen. The joint categorical
# policy is normalized only over executable nonzero in-bounds displacements.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-068-masked-64k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-068-masked-64k-v1-validation}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-65536}"
export NAVIGATOR_EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-160}"
export NAVIGATOR_SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-160}"
export NAVIGATOR_ACTION_DISTRIBUTION=masked_categorical

exec "$SCRIPT_DIR/run_navigator_bomopi_068_repaired.sh" "$@"
