#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/matey/project/segmentor-fix-navigator}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/checkpoints/navigator-bomopi-gru-filters-pretrained-development256k-v1/data/bomopi_resampled2_development15-v1/checkpoint_204800best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/navigator_bomopi/gru-filters-pretrained-development256k-v1-stochastic-best}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-navigator}"

exec uv run --no-sync python scripts/evaluate_navigator_stochastic.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --rollout-seed 101 \
  --rollout-seed 202 \
  --rollout-seed 303 \
  --expected-case-id pt1 \
  --expected-case-id pt11 \
  --expected-case-id pt6 \
  --selection stochastic
