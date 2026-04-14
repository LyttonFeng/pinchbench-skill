#!/usr/bin/env bash
# RunPod 环境初始化脚本
#
# 用法（RunPod 实例上执行）：
#   bash rl/scripts/setup_runpod.sh

set -euo pipefail

VERL_VERSION="${VERL_VERSION:-0.7.1}"
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"

echo "=============================="
echo "  RunPod 环境初始化"
echo "=============================="

# 基础依赖（Ubuntu PEP 668 需 --break-system-packages，与 flash/verl 一致）
pip install -q --break-system-packages \
    --no-cache-dir \
    "vllm==${VLLM_VERSION}" \
    transformers \
    peft \
    accelerate \
    datasets

# flash-attn: 必须使用预编译 wheel，禁止源码编译
# 当前 RunPod 常见环境是 torch 2.10.0+cu128 / Python 3.11。
# 这里优先走 mjun0812 的预编译 wheel，避免 GitHub release 链接失效后回退到源码编译。
PY_VER=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
if [[ "$PY_VER" == "cp312" ]]; then
    pip install -q --break-system-packages \
        --no-cache-dir \
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl"
elif [[ "$PY_VER" == "cp311" ]]; then
    pip install -q --break-system-packages \
        --no-cache-dir \
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.10-cp311-cp311-linux_x86_64.whl"
else
    echo "Unsupported Python version for flash-attn wheel: ${PY_VER}"
    exit 1
fi

# veRL
# Pin to the latest verified veRL release. It provides the importlib reward
# manager path used by turn-level rewards.
pip install -q --break-system-packages --no-cache-dir --upgrade "verl==${VERL_VERSION}"

echo ""
echo "验证 GPU..."
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| 设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "验证 vLLM..."
python -c "import vllm; print('vLLM 版本:', vllm.__version__)"

echo ""
echo "验证 flash-attn..."
python -c "import flash_attn; print('flash-attn 版本:', flash_attn.__version__)"

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
