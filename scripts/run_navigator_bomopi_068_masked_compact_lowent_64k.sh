#!/usr/bin/env bash
set -euo pipefail

# Categorical entropy-scale ablation. The original 0.001 coefficient produces
# a ~0.005 loss term over 156 categories, comparable to the PPO policy loss.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-068-masked-compact-lowent-64k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-068-masked-compact-lowent-64k-v1-validation}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-65536}"
export NAVIGATOR_ENT_COEF=0.0001

exec "$SCRIPT_DIR/run_navigator_bomopi_068_masked_compact_64k.sh" "$@"
