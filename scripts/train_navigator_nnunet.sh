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
TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-1000000}"
LEARNING_RATE="${NAVIGATOR_LEARNING_RATE:-0.00005}"
EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-400}"
SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-50}"
BATCH_SIZE="${NAVIGATOR_BATCH_SIZE:-128}"
UPDATE_EPOCHS="${NAVIGATOR_UPDATE_EPOCHS:-4}"

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
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --no-track-wandb \
  --track-tensorboard \
  --train-val-split 0.9 \
  --shuffle-dataset \
  --voxel-size-mm 1.5 \
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
  --batch-size "$BATCH_SIZE" \
  --update-epochs "$UPDATE_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --ent-coef 0.003 \
  --vf-coef 0.5 \
  --behavior-cloning-epochs 0 \
  --num-episodes-per-sample 2 \
  --num-steps-per-sample 2048 \
  --num-workers 1 \
  --save-freq "$SAVE_FREQ" \
  --eval-interval "$EVAL_INTERVAL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  "$@"
