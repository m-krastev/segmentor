#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/matey/project/segmentor-fix-navigator}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/bomopi_resampled2_unique-v1}"
RESULT_DIR="${RESULT_DIR:-/home/matey/project/segmentor/results/navigator_bomopi}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-/home/matey/project/segmentor/checkpoints/tensorboard}"
RUN_NAME="${RUN_NAME:-navigator-image-pretrain-bomopi-v1}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-navigator}"

exec uv run --no-sync python scripts/pretrain_navigator_image_encoder.py \
  --data-dir "${DATA_DIR}" \
  --train-case-id pt2 \
  --train-case-id pt4 \
  --train-case-id pt5 \
  --train-case-id pt7 \
  --train-case-id pt9 \
  --train-case-id pt12 \
  --train-case-id pt15 \
  --train-case-id pt17 \
  --train-case-id pt19 \
  --train-case-id pt20 \
  --train-case-id pt21 \
  --train-case-id pt23 \
  --validation-case-id pt1 \
  --validation-case-id pt11 \
  --validation-case-id pt6 \
  --patch-size-vox 32 \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --batches-per-case 256 \
  --validation-interval 250 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --mask-fraction 0.4 \
  --mask-cube-size 4 \
  --noise-std 0.03 \
  --seed 42 \
  --device cuda \
  --amp-dtype bf16 \
  --output "${RESULT_DIR}/${RUN_NAME}.pt" \
  --tensorboard-log-dir "${TENSORBOARD_ROOT}/${RUN_NAME}"
