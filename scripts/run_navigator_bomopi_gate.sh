#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
UV_BIN="${UV_BIN:-$(command -v uv)}"
SOURCE_DATA_DIR="${NAVIGATOR_SOURCE_DATA_DIR:-data/bomopi_resampled2}"
DATA_DIR="${NAVIGATOR_DATA_DIR:-data/bomopi_resampled2_unique-v1}"
MANIFEST="${NAVIGATOR_BOMOPI_MANIFEST:-experiments/navigator-bomopi-v1/eligible.txt}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-navigator}"
export UV_NO_PROGRESS="${UV_NO_PROGRESS:-1}"
export NAVIGATOR_DATA_DIR="$DATA_DIR"

"$UV_BIN" run --no-sync python -O scripts/prepare_navigator_bomopi_view.py \
  --source "$SOURCE_DATA_DIR" \
  --view "$DATA_DIR" \
  --manifest "$MANIFEST"

"$UV_BIN" run --no-sync python -O scripts/preflight_navigator_bomopi.py \
  --data-dir "$DATA_DIR"

exec "$PROJECT_ROOT/scripts/train_navigator_bomopi.sh" "$@"
