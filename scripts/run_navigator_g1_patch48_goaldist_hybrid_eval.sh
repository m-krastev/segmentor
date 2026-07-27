#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAVIGATOR_MODE_PROBE_UNIT="${NAVIGATOR_MODE_PROBE_UNIT:-navigator-g1-patch48-goaldist-hybrid-eval-v1}"
export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-g1-patch48-goaldist-hybrid-eval-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_nnunet/g1-patch48-goaldist-hybrid-eval-v1-validation}"
export NAVIGATOR_LOAD_FROM_CHECKPOINT="${NAVIGATOR_LOAD_FROM_CHECKPOINT:-checkpoints/navigator-g1-patch48-goaldist-bc1-probe-v1/nnunet-actual/behavior_cloning_model.pth}"
export NAVIGATOR_PATCH_SIZE_MM=48
export NAVIGATOR_OBSERVE_GOAL_DISTANCE=1
export NAVIGATOR_COVERAGE_GATED_GOAL_PLANNER=1
export NAVIGATOR_VALIDATION_MANIFEST="${NAVIGATOR_VALIDATION_MANIFEST:-experiments/navigator-nnunet-v2-validation-smoke3.txt}"

exec "$SCRIPT_DIR/run_navigator_g1_mode_probe.sh"
