#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${NAVIGATOR_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ACTION="${1:-status}"
UNIT="${2:-navigator-success-million}"
SERVICE="${UNIT%.service}.service"
UPSTREAM_UNIT="${3:-}"
TRAIN_SCRIPT="${NAVIGATOR_TRAIN_SCRIPT:-scripts/train_navigator_phantoms.sh}"

usage() {
  cat <<EOF
Usage: $0 {start|status|logs|stop|restart} [unit-name]
       $0 start-after [unit-name] upstream-unit

Examples:
  $0 start navigator-success-million
  $0 status navigator-success-million
  $0 logs navigator-success-million
  $0 stop navigator-success-million
  $0 start-after navigator-followup navigator-success-million

The start action runs scripts/train_navigator_phantoms.sh by default. Set
NAVIGATOR_TRAIN_SCRIPT=scripts/train_navigator_nnunet.sh for scratch training
on nnU-Net data. Override defaults with the NAVIGATOR_* variables documented
in the selected script.
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
    NAVIGATOR_VALIDATION_OUTPUT_DIR \
    NAVIGATOR_TOTAL_TIMESTEPS \
    NAVIGATOR_LEARNING_RATE \
    NAVIGATOR_LR_ANNEAL_TIMESTEPS \
    NAVIGATOR_EVAL_INTERVAL \
    NAVIGATOR_SAVE_FREQ \
    NAVIGATOR_NNUNET_RAW \
    NAVIGATOR_NNUNET_CACHE_DIR \
    NAVIGATOR_NNUNET_SEED_DIR \
    NAVIGATOR_NNUNET_CASE_IDS_FILE \
    NAVIGATOR_NNUNET_TRAIN_CASE_IDS_FILE \
    NAVIGATOR_NNUNET_VAL_CASE_IDS_FILE \
    NAVIGATOR_ENDPOINT_TOLERANCE_MM \
    NAVIGATOR_MAX_EPISODE_STEPS \
    NAVIGATOR_FRAMES_PER_BATCH \
    NAVIGATOR_GENERATE_EXPERT_PATH \
    NAVIGATOR_BEHAVIOR_CLONING_EPOCHS \
    NAVIGATOR_BC_BATCH_SIZE \
    NAVIGATOR_BC_MAX_POLICY_PROBABILITY \
    NAVIGATOR_BC_ACTION_STATISTIC \
    NAVIGATOR_PATCH_SIZE_MM \
    NAVIGATOR_OBSERVE_GOAL_DISTANCE \
    NAVIGATOR_OBSERVE_SEGMENTATION \
    NAVIGATOR_COVERAGE_GATED_GOAL_PLANNER \
    NAVIGATOR_PATH_RADIUS_MM \
    NAVIGATOR_GDT_REWARD_SCALE \
    NAVIGATOR_GDT_PROGRESS_NORMALIZATION \
    NAVIGATOR_TARGET_RECOVERY_REWARD_SCALE \
    NAVIGATOR_TARGET_DISTANCE_PENALTY_SCALE \
    NAVIGATOR_TARGET_DISTANCE_PENALTY_RADIUS_MM \
    NAVIGATOR_REVISIT_PENALTY_SCALE \
    NAVIGATOR_REWARD_CONTRACT \
    NAVIGATOR_GATE_POSITIVE_SHAPING_ON_TARGET_SEGMENT \
    NAVIGATOR_OFF_TARGET_PENALTY_SCALE \
    NAVIGATOR_WALL_PENALTY_SCALE \
    NAVIGATOR_TERMINAL_SUCCESS_BONUS \
    NAVIGATOR_TERMINAL_FAILURE_PENALTY \
    NAVIGATOR_ANNOTATION_FREE \
    NAVIGATOR_REWARD_SUPERVISED \
    NAVIGATOR_INTRINSIC_NOVELTY_REWARD_SCALE \
    NAVIGATOR_EPISODIC_CELL_REWARD_SCALE \
    NAVIGATOR_EPISODIC_CELL_SIZE_MM \
    NAVIGATOR_CURVATURE_PENALTY_SCALE \
    NAVIGATOR_FILTER_SCALES_MM \
    NAVIGATOR_TRAIN_VAL_SPLIT \
    NAVIGATOR_LOAD_FROM_CHECKPOINT \
    NAVIGATOR_RELOAD_CHECKPOINT_PATH \
    NAVIGATOR_EVAL_ONLY \
    NAVIGATOR_DETERMINISTIC_ACTION_STATISTIC \
    NAVIGATOR_MEMORY_MODEL \
    NAVIGATOR_MEMORY_HIDDEN_SIZE \
    NAVIGATOR_MEMORY_NUM_LAYERS \
    NAVIGATOR_S5_STATE_SIZE \
    NAVIGATOR_RECURRENT_SEQUENCE_LENGTH \
    NAVIGATOR_RECURRENT_BACKEND \
    NAVIGATOR_ACTION_DISTRIBUTION \
    NAVIGATOR_CATEGORICAL_ACTION_SUPPORT \
    NAVIGATOR_BATCH_SIZE \
    NAVIGATOR_UPDATE_EPOCHS \
    NAVIGATOR_ENT_COEF \
    NAVIGATOR_TARGET_KL \
    NAVIGATOR_VF_COEF \
    NAVIGATOR_MAX_GRAD_NORM \
    NAVIGATOR_SEPARATE_ACTOR_CRITIC_LOSSES \
    NAVIGATOR_REQUIRED_CHECKPOINT \
    NAVIGATOR_WAIT_POLL_SECONDS \
    UV_BIN
  do
    if [[ -n "${!variable:-}" ]]; then
      systemd_args+=("--setenv=$variable=${!variable}")
    fi
  done

  if [[ ! -x "$PROJECT_ROOT/$TRAIN_SCRIPT" ]]; then
    echo "Navigator training script is not executable: $PROJECT_ROOT/$TRAIN_SCRIPT" >&2
    exit 1
  fi

  if [[ "$ACTION" == "start-after" ]]; then
    if [[ -z "$UPSTREAM_UNIT" ]]; then
      echo "The start-after action requires an upstream unit." >&2
      exit 2
    fi
    systemd-run "${systemd_args[@]}" \
      /usr/bin/bash "$PROJECT_ROOT/scripts/wait_for_navigator_service.sh" \
      "$UPSTREAM_UNIT" "$PROJECT_ROOT/$TRAIN_SCRIPT"
  else
    systemd-run "${systemd_args[@]}" "$PROJECT_ROOT/$TRAIN_SCRIPT"
  fi
  echo "Follow logs with: $0 logs ${SERVICE%.service}"
}

case "$ACTION" in
  start)
    start_service
    ;;
  start-after)
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
