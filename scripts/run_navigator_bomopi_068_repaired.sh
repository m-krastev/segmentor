#!/usr/bin/env bash
set -euo pipefail

# Controlled repair of historical commit 068dc4d inside the current tested
# movement, recurrent PPO, validation, and telemetry implementation.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-068-repaired-256k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-068-repaired-256k-v1-validation}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-256000}"
export NAVIGATOR_EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-400}"
export NAVIGATOR_SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-50}"
export NAVIGATOR_PATCH_SIZE_MM="${NAVIGATOR_PATCH_SIZE_MM:-60}"
export NAVIGATOR_BATCH_SIZE="${NAVIGATOR_BATCH_SIZE:-32}"
export NAVIGATOR_ENT_COEF="${NAVIGATOR_ENT_COEF:-0.001}"
export NAVIGATOR_TARGET_KL="${NAVIGATOR_TARGET_KL:-0.03}"
export NAVIGATOR_OBSERVE_SEGMENTATION=false
export NAVIGATOR_ACTION_DISTRIBUTION=factorized_categorical
export NAVIGATOR_REWARD_CONTRACT=shin_normalized_repaired
export NAVIGATOR_POLICY_OBSERVATION_CONTRACT=shin_068_repaired

exec "$SCRIPT_DIR/run_navigator_bomopi_gate.sh" \
  --max-step-displacement-mm 9 \
  --cumulative-path-radius-mm 6 \
  --max-episode-steps 800 \
  --num-steps-per-sample 1600 \
  --frames-per-batch 1024 \
  --learning-rate 0.00001 \
  --gamma 0.99 \
  --batch-size 32 \
  --update-epochs 5 \
  "$@"
