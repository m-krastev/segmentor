#!/usr/bin/env bash
set -euo pipefail

# Foreground entry point for the successful phantom warm-start + PPO
# continuation. Environment variables below can override the run identity and
# schedule; additional CLI arguments are appended last.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
UV_BIN="${UV_BIN:-$(command -v uv)}"

CHECKPOINT="${NAVIGATOR_CHECKPOINT:-checkpoints/dagger-capped-v1/data/phantoms/behavior_cloning_model.pth}"
CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/dagger-ppo-million-v1}"
TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-1000000}"
LEARNING_RATE="${NAVIGATOR_LEARNING_RATE:-0.000005}"
EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-25}"
SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-25}"

cd "$PROJECT_ROOT"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Navigator checkpoint not found: $PROJECT_ROOT/$CHECKPOINT" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export UV_NO_PROGRESS="${UV_NO_PROGRESS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

exec "$UV_BIN" run --no-sync python -O -m navigator \
  --data-dir data/phantoms \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --no-track-wandb \
  --track-tensorboard \
  --load-from-checkpoint "$CHECKPOINT" \
  --voxel-size-mm 1 \
  --patch-size-mm 24 \
  --max-step-displacement-mm 6 \
  --cumulative-path-radius-mm 6 \
  --allowed-area-radius-mm 0 \
  --goal-action-prior 0 \
  --success-coverage-threshold 0.55 \
  --coverage-reward-scale 50 \
  --r-val2 1 \
  --r-zero-mov 1 \
  --max-episode-steps 1024 \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --frames-per-batch 1024 \
  --batch-size 128 \
  --update-epochs 1 \
  --learning-rate "$LEARNING_RATE" \
  --ent-coef 0.0005 \
  --behavior-cloning-epochs 0 \
  --num-steps-per-sample 8192 \
  --num-workers 1 \
  --save-freq "$SAVE_FREQ" \
  --eval-interval "$EVAL_INTERVAL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  "$@"
