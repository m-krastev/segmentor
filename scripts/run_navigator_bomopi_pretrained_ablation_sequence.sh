#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_SERVICE="${PRETRAIN_SERVICE:-navigator-image-pretrain-bomopi-v1.service}"

while systemctl --user is-active --quiet "${PRETRAIN_SERVICE}"; do
  sleep 10
done
if [[ "$(systemctl --user show "${PRETRAIN_SERVICE}" --property=Result --value)" != success ]]; then
  echo "Required image pretraining service failed: ${PRETRAIN_SERVICE}" >&2
  exit 1
fi

# Technical gate: verify checkpoint loading, six-channel expansion, and one
# complete training/validation cycle before spending the two matched 64k jobs.
NAVIGATOR_TOTAL_TIMESTEPS=4096 \
NAVIGATOR_EVAL_INTERVAL=8 \
NAVIGATOR_SAVE_FREQ=8 \
NAVIGATOR_CHECKPOINT_DIR=checkpoints/navigator-bomopi-gru-filters-pretrained-development-smoke-v1 \
NAVIGATOR_VALIDATION_OUTPUT_DIR=results/navigator_bomopi/gru-filters-pretrained-development-smoke-v1-validation \
  "${SCRIPT_DIR}/run_navigator_bomopi_filters_pretrained_development64k.sh"

# Matched seed-42 control: same 12/3 immutable split, PPO, reward, and state.
"${SCRIPT_DIR}/run_navigator_bomopi_filters_development64k.sh"

# The only intended difference is the label-free five-channel encoder
# checkpoint, safely expanded with a zero visitation-channel kernel.
"${SCRIPT_DIR}/run_navigator_bomopi_filters_pretrained_development64k.sh"
