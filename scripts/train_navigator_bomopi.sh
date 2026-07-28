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
  --use-immediate-gdt-reward \
  --terminate-on-success \
  --no-observe-goal-distance \
  --no-coverage-gated-goal-planner \
  --gate-positive-shaping-on-target-segment \
  --separate-actor-critic-losses \
  --memory-model gru \
  --memory-hidden-size 256 \
  --memory-num-layers 1 \
  --recurrent-sequence-length 64 \
  --recurrent-backend pad \
  --action-distribution factorized_categorical \
  --deterministic-action-statistic mode \
  --train-val-split 0.9 \
  --shuffle-dataset \
  --seed 42 \
  --voxel-size-mm 1.5 \
  --patch-size-mm 24 \
  --max-step-displacement-mm 6 \
  --cumulative-path-radius-mm 9 \
  --endpoint-tolerance-mm 3 \
  --allowed-area-radius-mm 0 \
  --goal-action-prior 0 \
  --success-coverage-threshold 0.40 \
  --coverage-reward-scale 50 \
  --gdt-reward-scale 0.1 \
  --gdt-progress-normalization max_step \
  --target-recovery-reward-scale 0.05 \
  --target-distance-penalty-scale 0.1 \
  --target-distance-penalty-radius-mm 600 \
  --step-penalty 0.01 \
  --wall-penalty-scale 0 \
  --r-val1 0 \
  --r-val2 1 \
  --r-zero-mov 1 \
  --terminal-success-bonus 50 \
  --terminal-failure-penalty 0 \
  --intrinsic-novelty-reward-scale 0 \
  --episodic-cell-reward-scale 0.01 \
  --episodic-cell-size-mm 6 \
  --curvature-penalty-scale 0 \
  --navigation-filter-scales-mm 3 6 9 \
  --max-episode-steps 2048 \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --frames-per-batch 1024 \
  --batch-size 128 \
  --update-epochs 4 \
  --learning-rate 0.00005 \
  --ent-coef 0.0005 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --num-episodes-per-sample 2 \
  --num-steps-per-sample 2048 \
  --num-workers 1 \
  --save-freq "$SAVE_FREQ" \
  --eval-interval "$EVAL_INTERVAL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  "$@"
