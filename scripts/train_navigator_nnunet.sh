#!/usr/bin/env bash
set -euo pipefail

# Foreground entry point for scratch PPO training on the complete intersection
# of the small-bowel, duodenum, and colon nnU-Net raw training cases.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
UV_BIN="${UV_BIN:-$(command -v uv)}"

NNUNET_RAW="${NAVIGATOR_NNUNET_RAW:-data/nnunet/nnUNet_raw}"
NNUNET_CACHE_DIR="${NAVIGATOR_NNUNET_CACHE_DIR:-results/navigator_nnunet/cache}"
NNUNET_SEED_DIR="${NAVIGATOR_NNUNET_SEED_DIR:-}"
CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-nnunet-scratch-v1}"
VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-}"
TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-1000000}"
LEARNING_RATE="${NAVIGATOR_LEARNING_RATE:-0.00005}"
EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-400}"
SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-50}"
BATCH_SIZE="${NAVIGATOR_BATCH_SIZE:-128}"
UPDATE_EPOCHS="${NAVIGATOR_UPDATE_EPOCHS:-4}"
ENT_COEF="${NAVIGATOR_ENT_COEF:-0.003}"
VF_COEF="${NAVIGATOR_VF_COEF:-0.5}"
MAX_GRAD_NORM="${NAVIGATOR_MAX_GRAD_NORM:-0.5}"
SEPARATE_ACTOR_CRITIC_LOSSES="${NAVIGATOR_SEPARATE_ACTOR_CRITIC_LOSSES:-0}"
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
TARGET_RECOVERY_REWARD_SCALE="${NAVIGATOR_TARGET_RECOVERY_REWARD_SCALE:-1}"
ANNOTATION_FREE="${NAVIGATOR_ANNOTATION_FREE:-1}"
REWARD_SUPERVISED="${NAVIGATOR_REWARD_SUPERVISED:-0}"
INTRINSIC_NOVELTY_REWARD_SCALE="${NAVIGATOR_INTRINSIC_NOVELTY_REWARD_SCALE:-0.05}"
CURVATURE_PENALTY_SCALE="${NAVIGATOR_CURVATURE_PENALTY_SCALE:-0.02}"
NAVIGATION_FILTER_SCALES_MM="${NAVIGATOR_FILTER_SCALES_MM:-3 6 9}"
TRAIN_VAL_SPLIT="${NAVIGATOR_TRAIN_VAL_SPLIT:-0.9}"
LOAD_FROM_CHECKPOINT="${NAVIGATOR_LOAD_FROM_CHECKPOINT:-}"
EVAL_ONLY="${NAVIGATOR_EVAL_ONLY:-0}"
DETERMINISTIC_ACTION_STATISTIC="${NAVIGATOR_DETERMINISTIC_ACTION_STATISTIC:-}"
MEMORY_MODEL="${NAVIGATOR_MEMORY_MODEL:-none}"
MEMORY_HIDDEN_SIZE="${NAVIGATOR_MEMORY_HIDDEN_SIZE:-256}"
MEMORY_NUM_LAYERS="${NAVIGATOR_MEMORY_NUM_LAYERS:-1}"
S5_STATE_SIZE="${NAVIGATOR_S5_STATE_SIZE:-256}"
RECURRENT_SEQUENCE_LENGTH="${NAVIGATOR_RECURRENT_SEQUENCE_LENGTH:-64}"
RECURRENT_BACKEND="${NAVIGATOR_RECURRENT_BACKEND:-pad}"
ACTION_DISTRIBUTION="${NAVIGATOR_ACTION_DISTRIBUTION:-beta}"

if [[ -z "$DETERMINISTIC_ACTION_STATISTIC" ]]; then
  if [[ "$ACTION_DISTRIBUTION" != "beta" ]]; then
    DETERMINISTIC_ACTION_STATISTIC="mode"
  else
    DETERMINISTIC_ACTION_STATISTIC="mean"
  fi
fi

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

LOSS_SEPARATION_ARGS=(--no-separate-actor-critic-losses)
if [[ "$SEPARATE_ACTOR_CRITIC_LOSSES" == "1" || "$SEPARATE_ACTOR_CRITIC_LOSSES" == "true" ]]; then
  LOSS_SEPARATION_ARGS=(--separate-actor-critic-losses)
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

ANNOTATION_ARGS=()
if [[ ("$ANNOTATION_FREE" == "1" || "$ANNOTATION_FREE" == "true") && \
      ("$REWARD_SUPERVISED" == "1" || "$REWARD_SUPERVISED" == "true") ]]; then
  echo "Annotation-free and reward-supervised modes are mutually exclusive." >&2
  exit 2
elif [[ "$ANNOTATION_FREE" == "1" || "$ANNOTATION_FREE" == "true" ]]; then
  if [[ -z "$NNUNET_SEED_DIR" ]]; then
    echo "NAVIGATOR_NNUNET_SEED_DIR is required for annotation-free training." >&2
    exit 2
  fi
  ANNOTATION_ARGS=(
    --annotation-free
    --no-reward-supervised
    --nnunet-seed-dir "$NNUNET_SEED_DIR"
    --no-use-immediate-gdt-reward
    --no-terminate-on-success
    --coverage-reward-scale 0
    --gdt-reward-scale 0
    --r-final 0
    --r-val1 0
  )
elif [[ "$REWARD_SUPERVISED" == "1" || "$REWARD_SUPERVISED" == "true" ]]; then
  if [[ -z "$NNUNET_SEED_DIR" ]]; then
    echo "NAVIGATOR_NNUNET_SEED_DIR is required for reward-supervised training." >&2
    exit 2
  fi
  ANNOTATION_ARGS=(
    --no-annotation-free
    --reward-supervised
    --nnunet-seed-dir "$NNUNET_SEED_DIR"
    --use-immediate-gdt-reward
    --terminate-on-success
    --coverage-reward-scale 50
    --gdt-reward-scale "$GDT_REWARD_SCALE"
    --target-recovery-reward-scale "$TARGET_RECOVERY_REWARD_SCALE"
    --r-final 50
    --r-val1 0.25
  )
else
  ANNOTATION_ARGS=(
    --no-annotation-free
    --no-reward-supervised
    --use-immediate-gdt-reward
    --terminate-on-success
    --coverage-reward-scale 50
    --gdt-reward-scale "$GDT_REWARD_SCALE"
    --target-recovery-reward-scale "$TARGET_RECOVERY_REWARD_SCALE"
    --r-final 50
    --r-val1 0.25
  )
fi

cd "$PROJECT_ROOT"
# The shared uv environment may have an editable install pointing at another
# checkout. Always execute the source tree selected by NAVIGATOR_PROJECT_ROOT.
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

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
  "${ANNOTATION_ARGS[@]}" \
  "${CASE_ARGS[@]}" \
  "${EXPERT_ARGS[@]}" \
  "${LOAD_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${GOAL_DISTANCE_ARGS[@]}" \
  "${GOAL_PLANNER_ARGS[@]}" \
  "${LOSS_SEPARATION_ARGS[@]}" \
  "${VALIDATION_ARGS[@]}" \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --no-track-wandb \
  --track-tensorboard \
  --deterministic-action-statistic "$DETERMINISTIC_ACTION_STATISTIC" \
  --memory-model "$MEMORY_MODEL" \
  --memory-hidden-size "$MEMORY_HIDDEN_SIZE" \
  --memory-num-layers "$MEMORY_NUM_LAYERS" \
  --s5-state-size "$S5_STATE_SIZE" \
  --recurrent-sequence-length "$RECURRENT_SEQUENCE_LENGTH" \
  --recurrent-backend "$RECURRENT_BACKEND" \
  --action-distribution "$ACTION_DISTRIBUTION" \
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
  --intrinsic-novelty-reward-scale "$INTRINSIC_NOVELTY_REWARD_SCALE" \
  --curvature-penalty-scale "$CURVATURE_PENALTY_SCALE" \
  --navigation-filter-scales-mm $NAVIGATION_FILTER_SCALES_MM \
  --r-val2 1 \
  --r-zero-mov 1 \
  --max-episode-steps "$MAX_EPISODE_STEPS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --frames-per-batch "$FRAMES_PER_BATCH" \
  --batch-size "$BATCH_SIZE" \
  --update-epochs "$UPDATE_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --ent-coef "$ENT_COEF" \
  --vf-coef "$VF_COEF" \
  --max-grad-norm "$MAX_GRAD_NORM" \
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
