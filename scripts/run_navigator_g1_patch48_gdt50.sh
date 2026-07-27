#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_DIAGNOSTIC_UNIT="${NAVIGATOR_DIAGNOSTIC_UNIT:-navigator-g1-patch48-gdt50-51k-v1}"
export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-g1-patch48-gdt50-51k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_nnunet/g1-patch48-gdt50-51k-v1-validation}"
export NAVIGATOR_LOAD_FROM_CHECKPOINT="${NAVIGATOR_LOAD_FROM_CHECKPOINT:-checkpoints/navigator-g1-patch48-bc1-probe-v1/nnunet-actual/behavior_cloning_model.pth}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-51200}"
export NAVIGATOR_FRAMES_PER_BATCH=256
export NAVIGATOR_BATCH_SIZE=16
export NAVIGATOR_UPDATE_EPOCHS=1
export NAVIGATOR_EVAL_INTERVAL=100
export NAVIGATOR_SAVE_FREQ=100
export NAVIGATOR_BEHAVIOR_CLONING_EPOCHS=0
export NAVIGATOR_GDT_REWARD_SCALE=50
export NAVIGATOR_DETERMINISTIC_ACTION_STATISTIC=mode
export NAVIGATOR_PATCH_SIZE_MM=48
export NAVIGATOR_VALIDATION_MANIFEST="${NAVIGATOR_VALIDATION_MANIFEST:-experiments/navigator-nnunet-v2-validation-smoke3.txt}"

exec "$SCRIPT_DIR/run_navigator_g1_diagnostic.sh"
