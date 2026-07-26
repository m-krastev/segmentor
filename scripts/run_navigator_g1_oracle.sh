#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
UV_BIN="${UV_BIN:-$(command -v uv)}"
UNIT="${NAVIGATOR_ORACLE_UNIT:-navigator-g1-oracle-validation-v1}"
MANIFEST_DIR="${NAVIGATOR_MANIFEST_DIR:-experiments/navigator-nnunet-v1}"
MANIFEST="$MANIFEST_DIR/validation.txt"
OUTPUT_DIR="${NAVIGATOR_ORACLE_OUTPUT_DIR:-results/navigator_nnunet/g1-oracle-validation-v1}"

cd "$PROJECT_ROOT"
if [[ ! -s "$MANIFEST" ]]; then
  echo "Missing immutable validation manifest: $PROJECT_ROOT/$MANIFEST" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to replace oracle output: $PROJECT_ROOT/$OUTPUT_DIR" >&2
  exit 1
fi

exec systemd-run \
  --user \
  "--unit=$UNIT" \
  --collect \
  --property=Type=exec \
  "--working-directory=$PROJECT_ROOT" \
  "--setenv=PYTHONPATH=$PROJECT_ROOT/src" \
  --setenv=UV_CACHE_DIR=/tmp/uv-cache-navigator \
  "$UV_BIN" run --no-sync python scripts/evaluate_navigator_oracle.py \
  --nnunet-raw data/nnunet/nnUNet_raw \
  --cache-dir results/navigator_nnunet/cache \
  --case-ids-file "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --oracle skeleton \
  --voxel-size-mm 1.5 \
  --patch-size-mm 24 \
  --max-episode-steps 2048 \
  --success-dice 0.40 \
  --path-radius-mm 9 \
  --endpoint-tolerance-mm 3
