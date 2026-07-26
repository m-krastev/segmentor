#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ACTION="${1:-status}"
UNIT="${2:-navigator-success-million}"
SERVICE="${UNIT%.service}.service"

usage() {
  cat <<EOF
Usage: $0 {start|status|logs|stop|restart} [unit-name]

Examples:
  $0 start navigator-success-million
  $0 status navigator-success-million
  $0 logs navigator-success-million
  $0 stop navigator-success-million

The start action runs scripts/train_navigator_phantoms.sh. Override its
defaults with NAVIGATOR_* environment variables documented in that script.
EOF
}

start_service() {
  if systemctl --user is-active --quiet "$SERVICE"; then
    echo "$SERVICE is already active." >&2
    exit 1
  fi

  local systemd_args=(
    --user
    "--unit=${SERVICE%.service}"
    --collect
    --property=Type=exec
    "--working-directory=$PROJECT_ROOT"
  )
  local variable
  for variable in \
    NAVIGATOR_CHECKPOINT \
    NAVIGATOR_CHECKPOINT_DIR \
    NAVIGATOR_TOTAL_TIMESTEPS \
    NAVIGATOR_LEARNING_RATE \
    NAVIGATOR_EVAL_INTERVAL \
    NAVIGATOR_SAVE_FREQ \
    UV_BIN
  do
    if [[ -n "${!variable:-}" ]]; then
      systemd_args+=("--setenv=$variable=${!variable}")
    fi
  done

  systemd-run "${systemd_args[@]}" "$PROJECT_ROOT/scripts/train_navigator_phantoms.sh"
  echo "Follow logs with: $0 logs ${SERVICE%.service}"
}

case "$ACTION" in
  start)
    start_service
    ;;
  status)
    systemctl --user status "$SERVICE" --no-pager
    ;;
  logs)
    exec journalctl --user -u "$SERVICE" -f
    ;;
  stop)
    systemctl --user stop "$SERVICE"
    ;;
  restart)
    systemctl --user stop "$SERVICE"
    start_service
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
