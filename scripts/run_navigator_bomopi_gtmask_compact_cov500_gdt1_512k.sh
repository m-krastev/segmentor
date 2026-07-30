#!/usr/bin/env bash
set -euo pipefail

# Exact-state continuation of the supervised 256k traversal champion.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-gtmask-compact-cov500-gdt1-512k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-gtmask-compact-cov500-gdt1-512k-v1-validation}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-512000}"
export NAVIGATOR_EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-1000}"
export NAVIGATOR_SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-500}"
export NAVIGATOR_RELOAD_CHECKPOINT_PATH="${NAVIGATOR_RELOAD_CHECKPOINT_PATH:-checkpoints/navigator-bomopi-gru-gtmask-compact-cov500-gdt1-256k-v1/data/bomopi_resampled2_unique-v1/checkpoint_256000.pth}"

exec "$SCRIPT_DIR/run_navigator_bomopi_gtmask_compact_cov500_gdt1_256k.sh" "$@"
