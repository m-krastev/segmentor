#!/usr/bin/env bash
set -euo pipefail

# Exact image-only control for the supervised perception upper bound. Keep the
# full navigation-filter state but remove only the GT segmentation channel.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-filters-compact-cov500-gdt1-256k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-filters-compact-cov500-gdt1-256k-v1-validation}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-256000}"
export NAVIGATOR_EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-1000}"
export NAVIGATOR_SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-500}"
export NAVIGATOR_ENT_COEF=0.0001
export NAVIGATOR_GDT_REWARD_SCALE=1.0
export NAVIGATOR_LR_ANNEAL_TIMESTEPS=65536
export NAVIGATOR_OBSERVE_SEGMENTATION=false
export NAVIGATOR_POLICY_OBSERVATION_CONTRACT=navigation_filters

exec "$SCRIPT_DIR/run_navigator_bomopi_compact_cov500_64k.sh" "$@"
