#!/usr/bin/env bash
set -euo pipefail

# Coverage-dominant follow-up selected by the exact BOMOPI action-signal
# audit. It retains the repaired-068 observation and compact masked action
# contracts while restoring a bounded cumulative-Dice potential and a
# geometrically feasible 2,048-step horizon.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-compact-cov500-64k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-compact-cov500-64k-v1-validation}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-65536}"
export NAVIGATOR_EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-160}"
export NAVIGATOR_SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-160}"
export NAVIGATOR_PATCH_SIZE_MM="${NAVIGATOR_PATCH_SIZE_MM:-60}"
export NAVIGATOR_BATCH_SIZE="${NAVIGATOR_BATCH_SIZE:-32}"
export NAVIGATOR_ENT_COEF="${NAVIGATOR_ENT_COEF:-0.001}"
export NAVIGATOR_TARGET_KL="${NAVIGATOR_TARGET_KL:-0.03}"
export NAVIGATOR_OBSERVE_SEGMENTATION="${NAVIGATOR_OBSERVE_SEGMENTATION:-false}"
export NAVIGATOR_ACTION_DISTRIBUTION=masked_categorical
export NAVIGATOR_CATEGORICAL_ACTION_SUPPORT=direction_length
export NAVIGATOR_REWARD_CONTRACT=potential
export NAVIGATOR_POLICY_OBSERVATION_CONTRACT="${NAVIGATOR_POLICY_OBSERVATION_CONTRACT:-shin_068_repaired}"
export NAVIGATOR_GDT_REWARD_SCALE="${NAVIGATOR_GDT_REWARD_SCALE:-0.1}"
export NAVIGATOR_COVERAGE_REWARD_SCALE=500
export NAVIGATOR_TARGET_DISTANCE_PENALTY_RADIUS_MM=60
export NAVIGATOR_TARGET_RECOVERY_REWARD_SCALE=0.2
export NAVIGATOR_EPISODIC_CELL_REWARD_SCALE=0
export NAVIGATOR_REVISIT_PENALTY_SCALE=0.01

exec "$SCRIPT_DIR/run_navigator_bomopi_gate.sh" \
  --max-step-displacement-mm 9 \
  --cumulative-path-radius-mm 6 \
  --max-episode-steps 2048 \
  --num-steps-per-sample 4096 \
  --frames-per-batch 512 \
  --learning-rate 0.00001 \
  --gamma 0.99 \
  --recurrent-sequence-length 32 \
  --batch-size 32 \
  --update-epochs 5 \
  "$@"
