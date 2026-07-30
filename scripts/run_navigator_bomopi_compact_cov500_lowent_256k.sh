#!/usr/bin/env bash
set -euo pipefail

# State-preserving concentration test after the coverage-dominant 64k screen.
# The only optimization change is a tenfold entropy-coefficient reduction.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-bomopi-gru-compact-cov500-lowent-256k-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_bomopi/gru-compact-cov500-lowent-256k-v1-validation}"
export NAVIGATOR_TOTAL_TIMESTEPS="${NAVIGATOR_TOTAL_TIMESTEPS:-256000}"
export NAVIGATOR_EVAL_INTERVAL="${NAVIGATOR_EVAL_INTERVAL:-640}"
export NAVIGATOR_SAVE_FREQ="${NAVIGATOR_SAVE_FREQ:-320}"
export NAVIGATOR_ENT_COEF=0.0001
export NAVIGATOR_LR_ANNEAL_TIMESTEPS=65536
export NAVIGATOR_RELOAD_CHECKPOINT_PATH="${NAVIGATOR_RELOAD_CHECKPOINT_PATH:-checkpoints/navigator-bomopi-gru-compact-cov500-64k-v1/data/bomopi_resampled2_unique-v1/final_model_torchrl.pth}"

exec "$SCRIPT_DIR/run_navigator_bomopi_compact_cov500_64k.sh" "$@"
