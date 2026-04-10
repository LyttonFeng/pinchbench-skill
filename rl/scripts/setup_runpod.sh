#!/usr/bin/env bash
# RunPod 环境初始化脚本
#
# 用法（RunPod 实例上执行）：
#   bash rl/scripts/setup_runpod.sh

set -euo pipefail

echo "=============================="
echo "  RunPod 环境初始化"
echo "=============================="

# 基础依赖
pip install -q \
    vllm \
    transformers \
    peft \
    accelerate \
    datasets

# veRL
# Pin the version that provides verl.experimental.agent_loop.AgentLoopBase,
# which OpenClawAgentLoop imports.
pip install -q --break-system-packages verl==0.7.1

echo ""
echo "验证 GPU..."
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| 设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "验证 vLLM..."
python -c "import vllm; print('vLLM 版本:', vllm.__version__)"

echo ""
echo "验证 veRL agent loop..."
python - <<'PY'
import verl
print('veRL 版本:', verl.__version__)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase
print('veRL experimental.agent_loop: OK')
PY

echo ""
echo "=============================="
echo "  环境初始化完成"
echo "  下一步: bash rl/scripts/start_vllm.sh Qwen/Qwen3-4B"
echo "=============================="
