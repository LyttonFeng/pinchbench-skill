#!/usr/bin/env bash
# RunPod 上启动 vLLM serve
#
# 用法：
#   bash rl/scripts/start_vllm.sh Qwen/Qwen3-4B
#   bash rl/scripts/start_vllm.sh Qwen/Qwen3-4B --port 8001
#
# 启动后 openclaw 配置：
#   --base-url http://<runpod-ip>:8000/v1
#   --model Qwen/Qwen3-4B

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-4B}"
PORT="${2:-8000}"
HOST="0.0.0.0"

echo "=============================="
echo "  启动 vLLM serve"
echo "  模型: ${MODEL}"
echo "  端口: ${PORT}"
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

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len 32768 \
    --trust-remote-code \
    --enable-prefix-caching \
    --dtype bfloat16
