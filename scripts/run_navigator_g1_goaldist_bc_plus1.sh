#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_DIAGNOSTIC_UNIT="${NAVIGATOR_DIAGNOSTIC_UNIT:-navigator-g1-goaldist-bc-plus1-probe-v1}"
export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-g1-goaldist-bc-plus1-probe-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_nnunet/g1-goaldist-bc-plus1-probe-v1-validation}"
export NAVIGATOR_LOAD_FROM_CHECKPOINT="${NAVIGATOR_LOAD_FROM_CHECKPOINT:-checkpoints/navigator-g1-goaldist-bc1-probe-v1/nnunet-actual/behavior_cloning_model.pth}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-256}"
export NAVIGATOR_FRAMES_PER_BATCH=256
export NAVIGATOR_BATCH_SIZE=16
export NAVIGATOR_BEHAVIOR_CLONING_EPOCHS=1
export NAVIGATOR_BC_BATCH_SIZE=64
export NAVIGATOR_BC_MAX_POLICY_PROBABILITY=0
export NAVIGATOR_BC_ACTION_STATISTIC=mean
export NAVIGATOR_DETERMINISTIC_ACTION_STATISTIC=mode
export NAVIGATOR_PATCH_SIZE_MM=24
export NAVIGATOR_OBSERVE_GOAL_DISTANCE=1
export NAVIGATOR_VALIDATION_MANIFEST="${NAVIGATOR_VALIDATION_MANIFEST:-experiments/navigator-nnunet-v2-validation-smoke3.txt}"

exec "$SCRIPT_DIR/run_navigator_g1_diagnostic.sh"
