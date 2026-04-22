#!/usr/bin/env bash
# Single-task online RL for task_18_spreadsheet_summary
# Reward = task18 event shaping + terminal reward only

set -euo pipefail

if [ -f "${HOME}/.pinchbench_env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${HOME}/.pinchbench_env"
  set +a
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK18_DATA_DIR="${TASK18_DATA_DIR:-${REPO_ROOT}/rl/data/prompts_task18_event_only}"
mkdir -p "${TASK18_DATA_DIR}"
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-/workspace/pinchbench-skill/task18_event_only_rl.log}"

python3 "${REPO_ROOT}/rl/train/prepare_prompts.py" \
  --tasks-dir "${REPO_ROOT}/tasks" \
  --output-dir "${TASK18_DATA_DIR}" \
  --task-ids task_18_spreadsheet_summary \
  --repeats "${TASK18_REPEATS:-8}"

export VERL_MODEL="${VERL_MODEL:-Qwen/Qwen3-1.7B}"
export REWARD_MODE="${REWARD_MODE:-task18-event-only}"
export PINCHBENCH_REWARD_MODULE_OVERRIDE="${PINCHBENCH_REWARD_MODULE_OVERRIDE:-${REPO_ROOT}/rl/agent_loop/task18_event_reward/reward_task18_event_only.py}"
export DATA_DIR="${TASK18_DATA_DIR}"
export PINCHBENCH_DATA_DIR_OVERRIDE="${TASK18_DATA_DIR}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MICRO_BATCH="${MICRO_BATCH:-1}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-4}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-16}"
export TEST_FREQ="${TEST_FREQ:-4}"
export SAVE_FREQ="${SAVE_FREQ:-4}"
export PINCHBENCH_REWARD_RETURN_MODE="${PINCHBENCH_REWARD_RETURN_MODE:-turn}"
export PINCHBENCH_TERMINAL_REWARD_WEIGHT="${PINCHBENCH_TERMINAL_REWARD_WEIGHT:-0.8}"
export PINCHBENCH_TASK_EMA_INIT="${PINCHBENCH_TASK_EMA_INIT:-0.0}"
export PINCHBENCH_TASK_EMA_ALPHA="${PINCHBENCH_TASK_EMA_ALPHA:-0.05}"
export MAX_TURNS="${MAX_TURNS:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-20000}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12000}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.28}"
export RUN_VERSION="${RUN_VERSION:-task18_event_only_qwen31}"
export OPENCLAW_HOST="${OPENCLAW_HOST:-8.163.82.224}"
export OPENCLAW_PORT="${OPENCLAW_PORT:-22}"
export OPENCLAW_USER="${OPENCLAW_USER:-root}"
export ECS_HOST="${ECS_HOST:-${OPENCLAW_HOST}}"

if ! python3 - <<'PY' >/dev/null 2>&1
from importlib.metadata import version
version("verl")
PY
then
  echo "ERROR: verl is not installed on this pod."
  echo "Fix:"
  echo "  cd ${REPO_ROOT}"
  echo "  ECS_HOST=${ECS_HOST} bash rl/scripts/setup_new_pod.sh"
  echo "Then rerun:"
  echo "  bash rl/train/run_reinforce_task18_event_only.sh"
  exit 1
fi

echo "=============================="
echo "task_18 event-only RL"
echo "MODEL=${VERL_MODEL}"
echo "DATA_DIR=${DATA_DIR}"
echo "REWARD_MODE=${REWARD_MODE}"
echo "REWARD_OVERRIDE=${PINCHBENCH_REWARD_MODULE_OVERRIDE}"
echo "TERMINAL_WEIGHT=${PINCHBENCH_TERMINAL_REWARD_WEIGHT}"
echo "EMA_INIT=${PINCHBENCH_TASK_EMA_INIT} EMA_ALPHA=${PINCHBENCH_TASK_EMA_ALPHA}"
echo "TOTAL_EPOCHS=${TOTAL_EPOCHS} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
echo "OPENCLAW=${OPENCLAW_USER}@${OPENCLAW_HOST}:${OPENCLAW_PORT}"
echo "LOG=${TRAIN_LOG_PATH}"
echo "=============================="

DATA_DIR="${DATA_DIR}" \
PINCHBENCH_DATA_DIR_OVERRIDE="${PINCHBENCH_DATA_DIR_OVERRIDE}" \
VERL_MODEL="${VERL_MODEL}" \
REWARD_MODE="${REWARD_MODE}" \
PINCHBENCH_REWARD_MODULE_OVERRIDE="${PINCHBENCH_REWARD_MODULE_OVERRIDE}" \
PINCHBENCH_REWARD_RETURN_MODE="${PINCHBENCH_REWARD_RETURN_MODE}" \
PINCHBENCH_TERMINAL_REWARD_WEIGHT="${PINCHBENCH_TERMINAL_REWARD_WEIGHT}" \
PINCHBENCH_TASK_EMA_INIT="${PINCHBENCH_TASK_EMA_INIT}" \
PINCHBENCH_TASK_EMA_ALPHA="${PINCHBENCH_TASK_EMA_ALPHA}" \
BATCH_SIZE="${BATCH_SIZE}" \
MICRO_BATCH="${MICRO_BATCH}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
TEST_FREQ="${TEST_FREQ}" \
SAVE_FREQ="${SAVE_FREQ}" \
MAX_TURNS="${MAX_TURNS}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL}" \
RUN_VERSION="${RUN_VERSION}" \
OPENCLAW_HOST="${OPENCLAW_HOST}" \
OPENCLAW_PORT="${OPENCLAW_PORT}" \
OPENCLAW_USER="${OPENCLAW_USER}" \
bash "${REPO_ROOT}/rl/train/run_reinforce_lora.sh"
