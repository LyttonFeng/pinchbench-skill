#!/usr/bin/env bash
# RunPod 上启动 vLLM serve
#
# 用法：
#   bash rl/scripts/start_vllm.sh Qwen/Qwen3-4B
#   VLLM_PORT=8001 bash rl/scripts/start_vllm.sh Qwen/Qwen3-4B
#
# LoRA 评测（adapter 目录含 adapter_model.safetensors）：
#   export LORA_ADAPTER_PATH=/path/to/global_step_25/actor/lora_adapter
#   export VLLM_LORA_NAME=pinchbench-lora   # API 里 --model 用这个名字
#   VLLM_PORT=8011 bash rl/scripts/start_vllm.sh Qwen/Qwen3-4B
#
# 启动后 openclaw / benchmark：
#   --base-url http://127.0.0.1:<PORT>/v1
#   基座模型请求: --model Qwen3-4B（与 --served-model-name 一致）
#   LoRA 请求: --model ${VLLM_LORA_NAME}

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-4B}"
PORT="${VLLM_PORT:-${2:-8000}}"
HOST="${VLLM_BIND_HOST:-0.0.0.0}"

echo "=============================="
echo "  启动 vLLM serve"
echo "  模型: ${MODEL}"
echo "  端口: ${PORT}"
if [[ -n "${LORA_ADAPTER_PATH:-}" ]]; then
  echo "  LoRA: ${VLLM_LORA_NAME:-pinchbench-lora} <- ${LORA_ADAPTER_PATH}"
fi
echo "=============================="

# 确认 vLLM 已安装
if ! python -c "import vllm" 2>/dev/null; then
    echo "[setup] 安装 vLLM..."
    pip install vllm -q
fi

# GPU 内存分配（Qwen3-4B 推理约 9GB，L4 24GB 够用）
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"

echo ""
echo "启动中，等待 ready..."
echo ""

LORA_NAME="${VLLM_LORA_NAME:-pinchbench-lora}"
LORA_RANK="${LORA_RANK:-32}"

CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "${MODEL}"
  --served-model-name "${MODEL}" "$(basename "${MODEL}")"
  --host "${HOST}"
  --port "${PORT}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-model-len 32768
  --trust-remote-code
  --enable-prefix-caching
  --dtype bfloat16
  --enable-auto-tool-choice
  --tool-call-parser hermes
  --reasoning-parser deepseek_r1
)

if [[ -n "${LORA_ADAPTER_PATH:-}" ]]; then
  if [[ ! -d "${LORA_ADAPTER_PATH}" ]]; then
    echo "ERROR: LORA_ADAPTER_PATH is not a directory: ${LORA_ADAPTER_PATH}"
    exit 1
  fi
  CMD+=(
    --enable-lora
    --max-loras 8
    --max-lora-rank "${LORA_RANK}"
    --lora-modules "${LORA_NAME}=${LORA_ADAPTER_PATH}"
  )
fi

exec "${CMD[@]}"
