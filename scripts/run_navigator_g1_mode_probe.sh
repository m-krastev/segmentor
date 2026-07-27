#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
UNIT="${NAVIGATOR_MODE_PROBE_UNIT:-navigator-g1-dagger-bc2-mode-probe-v1}"
OUTPUT_DIR="${NAVIGATOR_VALIDATION_OUTPUT_DIR:-results/navigator_nnunet/g1-dagger-bc2-mode-probe-v1-validation}"
CHECKPOINT="${NAVIGATOR_LOAD_FROM_CHECKPOINT:-checkpoints/navigator-g1-dagger-bc2-probe-v1/nnunet-actual/behavior_cloning_model.pth}"

cd "$PROJECT_ROOT"
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Missing behavior-cloning checkpoint: $PROJECT_ROOT/$CHECKPOINT" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to mix a mode probe with existing output: $PROJECT_ROOT/$OUTPUT_DIR" >&2
  exit 1
fi

export NAVIGATOR_TRAIN_SCRIPT=scripts/train_navigator_nnunet.sh
export NAVIGATOR_NNUNET_RAW="${NAVIGATOR_NNUNET_RAW:-data/nnunet/nnUNet_raw}"
export NAVIGATOR_NNUNET_CACHE_DIR="${NAVIGATOR_NNUNET_CACHE_DIR:-results/navigator_nnunet/cache}"
export NAVIGATOR_NNUNET_TRAIN_CASE_IDS_FILE="${NAVIGATOR_TRAIN_MANIFEST:-experiments/navigator-nnunet-v2/train.txt}"
export NAVIGATOR_NNUNET_VAL_CASE_IDS_FILE="${NAVIGATOR_VALIDATION_MANIFEST:-experiments/navigator-nnunet-v2-validation-smoke3.txt}"
export NAVIGATOR_CHECKPOINT_DIR="${NAVIGATOR_CHECKPOINT_DIR:-checkpoints/navigator-g1-dagger-bc2-mode-probe-v1}"
export NAVIGATOR_VALIDATION_OUTPUT_DIR="$OUTPUT_DIR"
export NAVIGATOR_LOAD_FROM_CHECKPOINT="$CHECKPOINT"
export NAVIGATOR_EVAL_ONLY=1
export NAVIGATOR_DETERMINISTIC_ACTION_STATISTIC=mode
export NAVIGATOR_GENERATE_EXPERT_PATH=1
export NAVIGATOR_MAX_EPISODE_STEPS="${NAVIGATOR_MAX_EPISODE_STEPS:-2048}"

exec "$SCRIPT_DIR/navigator_systemd.sh" start "$UNIT"
