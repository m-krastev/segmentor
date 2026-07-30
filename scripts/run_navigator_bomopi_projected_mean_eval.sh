#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/matey/project/segmentor-fix-navigator}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_DIR}/checkpoints}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/navigator_bomopi}"
DATA_SUFFIX="data/bomopi_resampled2_development15-v1"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-navigator}"

COMMON_ARGS=(
  --rollout-seed 42
  --expected-case-id pt1
  --expected-case-id pt11
  --expected-case-id pt6
  --selection mode
  --categorical-deterministic-decoding projected_mean
)

uv run --no-sync python scripts/evaluate_navigator_stochastic.py \
  --checkpoint "${CHECKPOINT_ROOT}/navigator-bomopi-gru-filters-development64k-v1/${DATA_SUFFIX}/checkpoint_49152best.pth" \
  --output-dir "${RESULT_ROOT}/gru-filters-development64k-v1-projected-mean-best" \
  "${COMMON_ARGS[@]}"

uv run --no-sync python scripts/evaluate_navigator_stochastic.py \
  --checkpoint "${CHECKPOINT_ROOT}/navigator-bomopi-gru-filters-pretrained-development64k-v1/${DATA_SUFFIX}/checkpoint_65536best.pth" \
  --output-dir "${RESULT_ROOT}/gru-filters-pretrained-development64k-v1-projected-mean-best" \
  "${COMMON_ARGS[@]}"
