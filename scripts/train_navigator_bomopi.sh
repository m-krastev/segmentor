#!/usr/bin/env bash
set -euo pipefail

# BOMOPI-only recurrent PPO gate. This intentionally uses the legacy
# per-subject directory loader and never opens the nnU-Net cohort.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
UV_BIN="${UV_BIN:-$(command -v uv)}"

DATA_DIR="${NAVIGATOR_DATA_DIR:-data/bomopi_resampled2_unique-v1}"
CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-102k-v1}"
VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-102k-v1-validation}"
TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-102400}"
EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-400}"
SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-50}"
PATCH_SIZE_MM="${NAVIGATOR_PATCH_SIZE_MM:-48}"
BATCH_SIZE="${NAVIGATOR_BATCH_SIZE:-64}"
GDT_REWARD_SCALE="${NAVIGATOR_GDT_REWARD_SCALE:-0.1}"
TARGET_DISTANCE_RADIUS_MM="${NAVIGATOR_TARGET_DISTANCE_PENALTY_RADIUS_MM:-600}"
TARGET_RECOVERY_REWARD_SCALE="${NAVIGATOR_TARGET_RECOVERY_REWARD_SCALE:-0.05}"
ENT_COEF="${NAVIGATOR_ENT_COEF:-0.0005}"
LR_ANNEAL_TIMESTEPS="${NAVIGATOR_LR_ANNEAL_TIMESTEPS:-0}"
TARGET_KL="${NAVIGATOR_TARGET_KL:-0}"
RELOAD_CHECKPOINT_PATH="${NAVIGATOR_RELOAD_CHECKPOINT_PATH:-}"
OBSERVE_SEGMENTATION="${NAVIGATOR_OBSERVE_SEGMENTATION:-false}"
ACTION_DISTRIBUTION="${NAVIGATOR_ACTION_DISTRIBUTION:-factorized_categorical}"
CATEGORICAL_ACTION_SUPPORT="${NAVIGATOR_CATEGORICAL_ACTION_SUPPORT:-dense}"
REVISIT_PENALTY_SCALE="${NAVIGATOR_REVISIT_PENALTY_SCALE:-0.01}"
REWARD_CONTRACT="${NAVIGATOR_REWARD_CONTRACT:-potential}"
POLICY_OBSERVATION_CONTRACT="${NAVIGATOR_POLICY_OBSERVATION_CONTRACT:-navigation_filters}"

cd "$PROJECT_ROOT"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "BOMOPI data directory not found: $PROJECT_ROOT/$DATA_DIR" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export UV_NO_PROGRESS="${UV_NO_PROGRESS:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-navigator}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

printf 'Navigator BOMOPI config: data=%s steps=%s patch_mm=%s batch=%s reward_contract=%s policy_observation_contract=%s potential_gdt_scale=%s potential_target_distance_radius_mm=%s potential_recovery_scale=%s ent_coef=%s lr_anneal_steps=%s target_kl=%s reload=%s observe_segmentation=%s action_distribution=%s categorical_action_support=%s potential_revisit_scale=%s\n' \
  "$DATA_DIR" \
  "$TOTAL_TIMESTEPS" \
  "$PATCH_SIZE_MM" \
  "$BATCH_SIZE" \
  "$REWARD_CONTRACT" \
  "$POLICY_OBSERVATION_CONTRACT" \
  "$GDT_REWARD_SCALE" \
  "$TARGET_DISTANCE_RADIUS_MM" \
  "$TARGET_RECOVERY_REWARD_SCALE" \
  "$ENT_COEF" \
  "$LR_ANNEAL_TIMESTEPS" \
  "$TARGET_KL" \
  "${RELOAD_CHECKPOINT_PATH:-none}" \
  "$OBSERVE_SEGMENTATION" \
  "$ACTION_DISTRIBUTION" \
  "$CATEGORICAL_ACTION_SUPPORT" \
  "$REVISIT_PENALTY_SCALE"

EXTRA_ARGS=()
if [[ -n "$RELOAD_CHECKPOINT_PATH" ]]; then
  EXTRA_ARGS+=(--reload-checkpoint-path "$RELOAD_CHECKPOINT_PATH")
fi
case "$OBSERVE_SEGMENTATION" in
  true|1|yes)
    EXTRA_ARGS+=(--observe-segmentation)
    ;;
  false|0|no)
    EXTRA_ARGS+=(--no-observe-segmentation)
    ;;
  *)
    echo "NAVIGATOR_OBSERVE_SEGMENTATION must be true or false" >&2
    exit 2
    ;;
esac

case "$REWARD_CONTRACT" in
  potential)
    REWARD_ARGS=(
      --use-immediate-gdt-reward
      --gate-positive-shaping-on-target-segment
      --coverage-reward-scale 50
      --gdt-reward-scale "$GDT_REWARD_SCALE"
      --gdt-progress-normalization max_step
      --target-recovery-reward-scale "$TARGET_RECOVERY_REWARD_SCALE"
      --target-distance-penalty-scale 0.1
      --target-distance-penalty-radius-mm "$TARGET_DISTANCE_RADIUS_MM"
      --step-penalty 0.01
      --revisit-penalty-scale "$REVISIT_PENALTY_SCALE"
      --wall-penalty-scale 0
      --r-val1 0
      --r-val2 1
      --r-zero-mov 1
      --terminal-success-bonus 50
      --terminal-failure-penalty 0
      --episodic-cell-reward-scale 0.01
    )
    ;;
  shin_normalized|shin_normalized_guarded|shin_normalized_repaired)
    # These legacy scales are disabled: the named contract supplies only the
    # fixed normalized Algorithm-1 terms implemented by the environment.
    REWARD_ARGS=(
      --no-use-immediate-gdt-reward
      --no-gate-positive-shaping-on-target-segment
      --coverage-reward-scale 0
      --gdt-reward-scale 0
      --target-recovery-reward-scale 0
      --target-distance-penalty-scale 0
      --step-penalty 0
      --revisit-penalty-scale 0
      --wall-penalty-scale 0
      --r-val1 0
      --r-val2 0
      --r-zero-mov 0
      --terminal-success-bonus 0
      --terminal-failure-penalty -1
      --episodic-cell-reward-scale 0
    )
    ;;
  *)
    echo \
      "NAVIGATOR_REWARD_CONTRACT must be potential, shin_normalized, shin_normalized_guarded, or shin_normalized_repaired" \
      >&2
    exit 2
    ;;
esac

exec "$UV_BIN" run --no-sync python -O -m navigator \
  --data-dir "$DATA_DIR" \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --no-track-wandb \
  --track-tensorboard \
  --validation-save-paths \
  --validation-output-dir "$VALIDATION_OUTPUT_DIR" \
  --no-annotation-free \
  --reward-supervised \
  --reward-contract "$REWARD_CONTRACT" \
  --policy-observation-contract "$POLICY_OBSERVATION_CONTRACT" \
  --terminate-on-success \
  --no-observe-goal-distance \
  --no-coverage-gated-goal-planner \
  --separate-actor-critic-losses \
  --memory-model gru \
  --memory-hidden-size 256 \
  --memory-num-layers 1 \
  --recurrent-sequence-length 64 \
  --recurrent-backend pad \
  --action-distribution "$ACTION_DISTRIBUTION" \
  --categorical-action-support "$CATEGORICAL_ACTION_SUPPORT" \
  --deterministic-action-statistic mode \
  --train-val-split 0.9 \
  --shuffle-dataset \
  --seed 42 \
  --voxel-size-mm 1.5 \
  --patch-size-mm "$PATCH_SIZE_MM" \
  --max-step-displacement-mm 6 \
  --cumulative-path-radius-mm 9 \
  --endpoint-tolerance-mm 3 \
  --allowed-area-radius-mm 0 \
  --goal-action-prior 0 \
  --success-coverage-threshold 0.40 \
  --intrinsic-novelty-reward-scale 0 \
  --episodic-cell-size-mm 6 \
  --curvature-penalty-scale 0 \
  --navigation-filter-scales-mm 3 6 9 \
  --max-episode-steps 2048 \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --frames-per-batch 1024 \
  --batch-size "$BATCH_SIZE" \
  --update-epochs 4 \
  --learning-rate 0.00005 \
  --lr-anneal-timesteps "$LR_ANNEAL_TIMESTEPS" \
  --ent-coef "$ENT_COEF" \
  --target-kl "$TARGET_KL" \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --num-episodes-per-sample 2 \
  --num-steps-per-sample 2048 \
  --num-workers 1 \
  --save-freq "$SAVE_FREQ" \
  --eval-interval "$EVAL_INTERVAL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  "${REWARD_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
