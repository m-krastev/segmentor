#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 upstream-unit command [argument ...]" >&2
  exit 2
fi

UPSTREAM="${1%.service}.service"
shift
POLL_SECONDS="${NAVIGATOR_WAIT_POLL_SECONDS:-30}"

while systemctl --user is-active --quiet "$UPSTREAM"; do
  sleep "$POLL_SECONDS"
done

if [[ -n "${NAVIGATOR_REQUIRED_CHECKPOINT:-}" &&
      ! -f "$NAVIGATOR_REQUIRED_CHECKPOINT" ]]; then
  echo "Required upstream checkpoint does not exist: $NAVIGATOR_REQUIRED_CHECKPOINT" >&2
  exit 1
fi

exec "$@"
