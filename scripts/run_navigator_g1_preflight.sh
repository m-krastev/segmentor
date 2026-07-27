#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
UV_BIN="${UV_BIN:-$(command -v uv)}"
UNIT="${NAVIGATOR_PREFLIGHT_UNIT:-navigator-preflight-v2}"
NNUNET_RAW="${NAVIGATOR_NNUNET_RAW:-data/nnunet/nnUNet_raw}"
CACHE_DIR="${NAVIGATOR_NNUNET_CACHE_DIR:-results/navigator_nnunet/cache}"
OUTPUT_DIR="${NAVIGATOR_MANIFEST_DIR:-experiments/navigator-nnunet-v2}"
MINIMUM_EXPERT_ROUTE_LENGTH_MM="${NAVIGATOR_MINIMUM_EXPERT_ROUTE_LENGTH_MM:-300}"

cd "$PROJECT_ROOT"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to replace immutable preflight output: $PROJECT_ROOT/$OUTPUT_DIR" >&2
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
  "$UV_BIN" run --no-sync python scripts/preflight_navigator_nnunet.py \
  --nnunet-raw "$NNUNET_RAW" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --seed 42 \
  --train-fraction 0.8 \
  --validation-fraction 0.1 \
  --minimum-expert-route-length-mm "$MINIMUM_EXPERT_ROUTE_LENGTH_MM" \
  --generate-expert-path
