#!/usr/bin/env bash
# 全量 PinchBench：先基座 Qwen3-4B，再 global_step_25 LoRA；生成对比表（标注 RL 训练 8 任务）。
#
# 前置（必做）：
#   1) 同一 GPU 上不要和 veRL 训练同时跑，否则会 OOM。先停训练：
#        tmux kill-session -t train
#   2) 本机已装 openclaw；~/.pinchbench_env 里有 DASHSCOPE_API_KEY（判分用）
#
# 用法：
#   cd /workspace/pinchbench-skill
#   bash rl/scripts/run_pinchbench_full_compare.sh
#
# 可选环境变量：
#   CKPT_LORA_ADAPTER  默认 REPO/rl/checkpoints/reinforce_lora/global_step_25/actor/lora_adapter
#   BASE_PORT / LORA_PORT  默认 8010 / 8011
#   BENCH_ALLOW_TRAINING=1  跳过「必须停训练」检查（不推荐）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "${HOME}/.pinchbench_env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${HOME}/.pinchbench_env"
  set +a
fi

CKPT_LORA_ADAPTER="${CKPT_LORA_ADAPTER:-${REPO_ROOT}/rl/checkpoints/reinforce_lora/global_step_25/actor/lora_adapter}"
BASE_PORT="${BASE_PORT:-8010}"
LORA_PORT="${LORA_PORT:-8011}"
STAMP="$(date +%Y%m%d_%H%M)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/bench_compare_${STAMP}}"
BASE_URL_B="http://127.0.0.1:${BASE_PORT}/v1"
BASE_URL_L="http://127.0.0.1:${LORA_PORT}/v1"

if [[ "${BENCH_ALLOW_TRAINING:-0}" != "1" ]] && pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1; then
  echo "ERROR: 检测到 main_ppo 训练仍在跑。请先停训练（例如 tmux kill-session -t train），"
  echo "       或设置 BENCH_ALLOW_TRAINING=1 强行继续（极易 OOM）。"
  exit 1
fi

if [[ ! -d "${CKPT_LORA_ADAPTER}" ]] || [[ ! -f "${CKPT_LORA_ADAPTER}/adapter_model.safetensors" ]]; then
  echo "ERROR: LoRA 目录无效: ${CKPT_LORA_ADAPTER}"
  exit 1
fi

wait_vllm_ready() {
  local port="$1"
  local name="$2"
  echo "[wait] ${name} http://127.0.0.1:${port}/v1/models ..."
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[wait] ${name} ready."
      return 0
    fi
    sleep 2
  done
  echo "ERROR: ${name} vLLM 未在 240s 内就绪"
  return 1
}

kill_pid() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
  sleep 3
}

mkdir -p "${OUT_ROOT}/baseline" "${OUT_ROOT}/lora"

echo "=============================="
echo "  OUT_ROOT=${OUT_ROOT}"
echo "  LoRA adapter: ${CKPT_LORA_ADAPTER}"
echo "=============================="

# ── Baseline vLLM ──
unset LORA_ADAPTER_PATH || true
export VLLM_PORT="${BASE_PORT}"
bash "${REPO_ROOT}/rl/scripts/start_vllm.sh" "Qwen/Qwen3-4B" &
PID_BASE=$!
wait_vllm_ready "${BASE_PORT}" "baseline"

(
  cd "${REPO_ROOT}/scripts"
  python3 benchmark.py \
    --model "Qwen3-4B" \
    --base-url "${BASE_URL_B}" \
    --api-key dummy \
    --suite all \
    --output-dir "${OUT_ROOT}/baseline" \
    --no-upload \
    --no-fail-fast \
    2>&1 | tee "${OUT_ROOT}/baseline_run.log"
)

kill_pid "${PID_BASE}"

# ── LoRA vLLM ──
export LORA_ADAPTER_PATH="${CKPT_LORA_ADAPTER}"
export VLLM_LORA_NAME="${VLLM_LORA_NAME:-pinchbench-lora}"
export VLLM_PORT="${LORA_PORT}"
bash "${REPO_ROOT}/rl/scripts/start_vllm.sh" "Qwen/Qwen3-4B" &
PID_LORA=$!
wait_vllm_ready "${LORA_PORT}" "lora"

(
  cd "${REPO_ROOT}/scripts"
  python3 benchmark.py \
    --model "${VLLM_LORA_NAME}" \
    --base-url "${BASE_URL_L}" \
    --api-key dummy \
    --suite all \
    --output-dir "${OUT_ROOT}/lora" \
    --no-upload \
    --no-fail-fast \
    2>&1 | tee "${OUT_ROOT}/lora_run.log"
)

kill_pid "${PID_LORA}"

BASE_JSON="$(ls -t "${OUT_ROOT}"/baseline/*.json 2>/dev/null | head -1)"
LORA_JSON="$(ls -t "${OUT_ROOT}"/lora/*.json 2>/dev/null | head -1)"
if [[ -z "${BASE_JSON}" ]] || [[ -z "${LORA_JSON}" ]]; then
  echo "ERROR: 未找到 benchmark 输出 JSON。请查 ${OUT_ROOT}/baseline_run.log"
  exit 1
fi

python3 "${REPO_ROOT}/rl/scripts/compare_benchmark_json.py" \
  "${BASE_JSON}" "${LORA_JSON}" \
  --out-dir "${OUT_ROOT}/summary"

echo ""
echo "完成。对比结果: ${OUT_ROOT}/summary/compare.md / compare.json"
echo "原始结果: ${BASE_JSON}"
echo "          ${LORA_JSON}"
