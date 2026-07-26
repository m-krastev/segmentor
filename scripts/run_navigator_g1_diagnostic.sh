#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
MANIFEST_DIR="${NAVIGATOR_MANIFEST_DIR:-experiments/navigator-nnunet-v1}"
UNIT="${NAVIGATOR_DIAGNOSTIC_UNIT:-navigator-g1-diagnostic-250k}"
CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-g1-diagnostic-250k}"
VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_nnunet/g1-diagnostic-250k-validation}"

cd "$PROJECT_ROOT"
for manifest in train validation; do
  if [[ ! -s "$MANIFEST_DIR/$manifest.txt" ]]; then
    echo "Missing immutable $manifest manifest: $PROJECT_ROOT/$MANIFEST_DIR/$manifest.txt" >&2
    exit 1
  fi
done
for output in "$CHECKPOINT_DIR" "$VALIDATION_OUTPUT_DIR"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to mix a new diagnostic with existing output: $PROJECT_ROOT/$output" >&2
    exit 1
  fi
done

export NAVIGATOR_TRAIN_SCRIPT=scripts/train_navigator_nnunet.sh
export NAVIGATOR_NNUNET_RAW="${NAVIGATOR_NNUNET_RAW:-data/nnunet/nnUNet_raw}"
export NAVIGATOR_NNUNET_CACHE_DIR="${NAVIGATOR_NNUNET_CACHE_DIR:-results/navigator_nnunet/cache}"
export NAVIGATOR_NNUNET_TRAIN_CASE_IDS_FILE="$MANIFEST_DIR/train.txt"
export NAVIGATOR_NNUNET_VAL_CASE_IDS_FILE="$MANIFEST_DIR/validation.txt"
export NAVIGATOR_CHECKPOINT_DIR="$CHECKPOINT_DIR"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="$VALIDATION_OUTPUT_DIR"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-250000}"
export NAVIGATOR_GENERATE_EXPERT_PATH=1
export NAVIGATOR_BEHAVIOR_CLONING_EPOCHS="${NAVIGATOR_BEHAVIOR_CLONING_EPOCHS:-1}"
export NAVIGATOR_BC_MAX_POLICY_PROBABILITY="${NAVIGATOR_BC_MAX_POLICY_PROBABILITY:-0}"
export NAVIGATOR_PATH_RADIUS_MM=9
export NAVIGATOR_ENDPOINT_TOLERANCE_MM=3
export NAVIGATOR_MAX_EPISODE_STEPS="${NAVIGATOR_MAX_EPISODE_STEPS:-2048}"
export NAVIGATOR_FRAMES_PER_BATCH=1024
export NAVIGATOR_BATCH_SIZE="${NAVIGATOR_BATCH_SIZE:-128}"
export NAVIGATOR_UPDATE_EPOCHS="${NAVIGATOR_UPDATE_EPOCHS:-1}"
export NAVIGATOR_LEARNING_RATE="${NAVIGATOR_LEARNING_RATE:-0.00005}"
export NAVIGATOR_GDT_REWARD_SCALE="${NAVIGATOR_GDT_REWARD_SCALE:-1}"
export NAVIGATOR_EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-25}"
export NAVIGATOR_SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-25}"

exec scripts/navigator_systemd.sh start "$UNIT"
