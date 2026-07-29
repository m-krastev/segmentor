#!/usr/bin/env bash
set -euo pipefail

# Registered continuation of the completed compact 64k screen. Preserve its
# completed 64k cosine schedule so resume cannot raise the learning rate again.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-068-masked-compact-256k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-068-masked-compact-256k-v1-validation}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-256000}"
export NAVIGATOR_LR_ANNEAL_TIMESTEPS=65536
export NAVIGATOR_RELOAD_CHECKPOINT_PATH="${NAVIGATOR_RELOAD_CHECKPOINT_PATH:-checkpoints/navigator-bomopi-gru-068-masked-compact-64k-v1/data/bomopi_resampled2_unique-v1/final_model_torchrl.pth}"

exec "$SCRIPT_DIR/run_navigator_bomopi_068_masked_compact_64k.sh" "$@"
