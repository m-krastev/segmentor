#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_VISUAL_ENCODER_CHECKPOINT="${NAVIGATOR_VISUAL_ENCODER_CHECKPOINT:-/home/matey/project/segmentor/results/navigator_bomopi/navigator-image-pretrain-bomopi-v1.pt}"
export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-filters-pretrained-development64k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-filters-pretrained-development64k-v1-validation}"

exec "$SCRIPT_DIR/run_navigator_bomopi_filters_development64k.sh" "$@"
