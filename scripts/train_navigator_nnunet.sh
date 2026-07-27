#!/usr/bin/env bash
set -euo pipefail

# Foreground entry point for scratch PPO training on the complete intersection
# of the small-bowel, duodenum, and colon nnU-Net raw training cases.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
UV_BIN="${UV_BIN:-$(command -v uv)}"

NNUNET_RAW="${NAVIGATOR_NNUNET_RAW:-data/nnunet/nnUNet_raw}"
NNUNET_CACHE_DIR="${NAVIGATOR_NNUNET_CACHE_DIR:-results/navigator_nnunet/cache}"
CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-nnunet-scratch-v1}"
VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-}"
TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-1000000}"
LEARNING_RATE="${NAVIGATOR_LEARNING_RATE:-0.00005}"
EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-400}"
SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-50}"
BATCH_SIZE="${NAVIGATOR_BATCH_SIZE:-128}"
UPDATE_EPOCHS="${NAVIGATOR_UPDATE_EPOCHS:-4}"
CASE_IDS_FILE="${NAVIGATOR_NNUNET_CASE_IDS_FILE:-}"
TRAIN_CASE_IDS_FILE="${NAVIGATOR_NNUNET_TRAIN_CASE_IDS_FILE:-}"
VAL_CASE_IDS_FILE="${NAVIGATOR_NNUNET_VAL_CASE_IDS_FILE:-}"
GENERATE_EXPERT_PATH="${NAVIGATOR_GENERATE_EXPERT_PATH:-0}"
BEHAVIOR_CLONING_EPOCHS="${NAVIGATOR_BEHAVIOR_CLONING_EPOCHS:-0}"
BC_BATCH_SIZE="${NAVIGATOR_BC_BATCH_SIZE:-64}"
BC_MAX_POLICY_PROBABILITY="${NAVIGATOR_BC_MAX_POLICY_PROBABILITY:-1}"
BC_ACTION_STATISTIC="${NAVIGATOR_BC_ACTION_STATISTIC:-mean}"
PATCH_SIZE_MM="${NAVIGATOR_PATCH_SIZE_MM:-24}"
OBSERVE_GOAL_DISTANCE="${NAVIGATOR_OBSERVE_GOAL_DISTANCE:-0}"
COVERAGE_GATED_GOAL_PLANNER="${NAVIGATOR_COVERAGE_GATED_GOAL_PLANNER:-0}"
PATH_RADIUS_MM="${NAVIGATOR_PATH_RADIUS_MM:-9}"
ENDPOINT_TOLERANCE_MM="${NAVIGATOR_ENDPOINT_TOLERANCE_MM:-3}"
MAX_EPISODE_STEPS="${NAVIGATOR_MAX_EPISODE_STEPS:-2048}"
FRAMES_PER_BATCH="${NAVIGATOR_FRAMES_PER_BATCH:-1024}"
GDT_REWARD_SCALE="${NAVIGATOR_GDT_REWARD_SCALE:-1}"
TRAIN_VAL_SPLIT="${NAVIGATOR_TRAIN_VAL_SPLIT:-0.9}"
LOAD_FROM_CHECKPOINT="${NAVIGATOR_LOAD_FROM_CHECKPOINT:-}"
EVAL_ONLY="${NAVIGATOR_EVAL_ONLY:-0}"
DETERMINISTIC_ACTION_STATISTIC="${NAVIGATOR_DETERMINISTIC_ACTION_STATISTIC:-mean}"

CASE_ARGS=()
if [[ -n "$CASE_IDS_FILE" ]]; then
  CASE_ARGS=(--nnunet-case-ids-file "$CASE_IDS_FILE")
fi
if [[ -n "$TRAIN_CASE_IDS_FILE" || -n "$VAL_CASE_IDS_FILE" ]]; then
  if [[ -z "$TRAIN_CASE_IDS_FILE" || -z "$VAL_CASE_IDS_FILE" ]]; then
    echo "Both NAVIGATOR_NNUNET_TRAIN_CASE_IDS_FILE and NAVIGATOR_NNUNET_VAL_CASE_IDS_FILE are required." >&2
    exit 2
  fi
  if [[ -n "$CASE_IDS_FILE" ]]; then
    echo "Explicit train/validation manifests cannot be combined with NAVIGATOR_NNUNET_CASE_IDS_FILE." >&2
    exit 2
  fi
  CASE_ARGS=(
    --nnunet-train-case-ids-file "$TRAIN_CASE_IDS_FILE"
    --nnunet-val-case-ids-file "$VAL_CASE_IDS_FILE"
  )
fi

EXPERT_ARGS=()
if [[ "$GENERATE_EXPERT_PATH" == "1" || "$GENERATE_EXPERT_PATH" == "true" ]]; then
  EXPERT_ARGS=(--nnunet-generate-expert-path)
fi

LOAD_ARGS=()
if [[ -n "$LOAD_FROM_CHECKPOINT" ]]; then
  LOAD_ARGS=(--load-from-checkpoint "$LOAD_FROM_CHECKPOINT")
fi

EVAL_ARGS=()
if [[ "$EVAL_ONLY" == "1" || "$EVAL_ONLY" == "true" ]]; then
  EVAL_ARGS=(--eval-only)
fi

GOAL_DISTANCE_ARGS=(--no-observe-goal-distance)
if [[ "$OBSERVE_GOAL_DISTANCE" == "1" || "$OBSERVE_GOAL_DISTANCE" == "true" ]]; then
  GOAL_DISTANCE_ARGS=(--observe-goal-distance)
fi

GOAL_PLANNER_ARGS=(--no-coverage-gated-goal-planner)
if [[ "$COVERAGE_GATED_GOAL_PLANNER" == "1" || "$COVERAGE_GATED_GOAL_PLANNER" == "true" ]]; then
  GOAL_PLANNER_ARGS=(--coverage-gated-goal-planner)
fi

VALIDATION_ARGS=()
if [[ -n "$VALIDATION_OUTPUT_DIR" ]]; then
  VALIDATION_ARGS=(--validation-output-dir "$VALIDATION_OUTPUT_DIR")
fi

cd "$PROJECT_ROOT"

if [[ ! -d "$NNUNET_RAW" ]]; then
  echo "nnU-Net raw directory not found: $PROJECT_ROOT/$NNUNET_RAW" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export UV_NO_PROGRESS="${UV_NO_PROGRESS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

exec "$UV_BIN" run --no-sync python -O -m navigator \
  --data-dir nnunet-actual \
  --nnunet-raw-dir "$NNUNET_RAW" \
  --nnunet-cache-dir "$NNUNET_CACHE_DIR" \
  "${CASE_ARGS[@]}" \
  "${EXPERT_ARGS[@]}" \
  "${LOAD_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${GOAL_DISTANCE_ARGS[@]}" \
  "${GOAL_PLANNER_ARGS[@]}" \
  "${VALIDATION_ARGS[@]}" \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --no-track-wandb \
  --track-tensorboard \
  --deterministic-action-statistic "$DETERMINISTIC_ACTION_STATISTIC" \
  --train-val-split "$TRAIN_VAL_SPLIT" \
  --shuffle-dataset \
  --voxel-size-mm 1.5 \
  --patch-size-mm "$PATCH_SIZE_MM" \
  --max-step-displacement-mm 6 \
  --cumulative-path-radius-mm "$PATH_RADIUS_MM" \
  --endpoint-tolerance-mm "$ENDPOINT_TOLERANCE_MM" \
  --allowed-area-radius-mm 0 \
  --goal-action-prior 0 \
  --success-coverage-threshold 0.40 \
  --coverage-reward-scale 50 \
  --gdt-reward-scale "$GDT_REWARD_SCALE" \
  --r-val2 1 \
  --r-zero-mov 1 \
  --max-episode-steps "$MAX_EPISODE_STEPS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --frames-per-batch "$FRAMES_PER_BATCH" \
  --batch-size "$BATCH_SIZE" \
  --update-epochs "$UPDATE_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --ent-coef 0.003 \
  --vf-coef 0.5 \
  --behavior-cloning-epochs "$BEHAVIOR_CLONING_EPOCHS" \
  --behavior-cloning-batch-size "$BC_BATCH_SIZE" \
  --behavior-cloning-max-policy-probability "$BC_MAX_POLICY_PROBABILITY" \
  --behavior-cloning-action-statistic "$BC_ACTION_STATISTIC" \
  --num-episodes-per-sample 2 \
  --num-steps-per-sample 2048 \
  --num-workers 1 \
  --save-freq "$SAVE_FREQ" \
  --eval-interval "$EVAL_INTERVAL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  "$@"
