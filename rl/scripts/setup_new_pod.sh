#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  RunPod 新 Pod 一键初始化脚本
#
#  镜像要求: PyTorch 2.10+ / CUDA 12.8 / Python 3.11
#  推荐镜像: runpod/pytorch:2.10.0-py3.11-cuda12.8.0-devel-ubuntu22.04
#  GPU: NVIDIA L40S (48GB)
#
#  用法:
#    1. 新建 Pod，SSH 进去
#    2. 运行: bash setup_new_pod.sh
#    3. 按提示启动训练
#
#  ECS 信息 (阿里云 4核8G):
#    IP:   8.163.82.224
#    User: root
#    Port: 22
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

ECS_HOST="8.163.82.224"
ECS_USER="root"
ECS_PORT=22
REPO_URL="https://github.com/LyttonFeng/pinchbench-skill.git"
REPO_DIR="/workspace/pinchbench-skill"
HF_CACHE="/workspace/hf_cache"
MODEL="Qwen/Qwen3-4B"

echo "══════════════════════════════════════"
echo "  RunPod 新 Pod 初始化"
echo "  ECS: ${ECS_USER}@${ECS_HOST}:${ECS_PORT}"
echo "══════════════════════════════════════"

# ── 1. SSH Key ──
echo ""
echo "[1/6] 配置 SSH Key..."
if [ ! -f /root/.ssh/id_ed25519 ]; then
    ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N "" -q
    echo "✓ 生成新 SSH key"
fi
PUB_KEY=$(cat /root/.ssh/id_ed25519.pub)
echo "公钥: ${PUB_KEY}"
echo ""
echo ">>> 将此公钥添加到 ECS authorized_keys <<<"
echo ">>> 运行 (从你的本地机器):"
echo "    ssh root@${ECS_HOST} \"echo '${PUB_KEY}' >> ~/.ssh/authorized_keys\""
echo ""
read -p "已添加公钥到 ECS? [y/N] " -r
if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "请先添加公钥，然后重新运行此脚本"
    exit 1
fi

echo "测试 SSH 连接..."
if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i /root/.ssh/id_ed25519 \
    ${ECS_USER}@${ECS_HOST} -p ${ECS_PORT} "echo 'SSH OK'"; then
    echo "✓ SSH 连接成功"
else
    echo "✗ SSH 连接失败，请检查公钥是否正确添加"
    exit 1
fi

# ── 2. 安装依赖 ──
echo ""
echo "[2/6] 安装 Python 依赖..."
pip install -q \
    verl==0.7.1 \
    vllm==0.19.0 \
    transformers \
    peft \
    accelerate \
    datasets \
    aiohttp

echo "✓ 依赖安装完成"

# ── 3. 克隆代码 ──
echo ""
echo "[3/6] 克隆代码..."
if [ -d "${REPO_DIR}" ]; then
    echo "仓库已存在，pull 最新代码..."
    cd "${REPO_DIR}" && git stash 2>/dev/null || true
    git pull
else
    git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
echo "✓ 代码: $(git log --oneline -1)"

# ── 4. 预下载模型 ──
echo ""
echo "[4/6] 预下载模型 ${MODEL}..."
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('下载 tokenizer...')
AutoTokenizer.from_pretrained('${MODEL}', cache_dir='${HF_CACHE}')
print('下载模型权重...')
AutoModelForCausalLM.from_pretrained('${MODEL}', cache_dir='${HF_CACHE}', torch_dtype='auto')
print('✓ 模型下载完成')
"

# ── 5. 准备训练数据 ──
echo ""
echo "[5/6] 准备训练数据..."
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
if [ ! -f rl/data/prompts/train.parquet ]; then
    python3 rl/train/prepare_prompts.py --tasks-dir tasks/ --output-dir rl/data/prompts/
    echo "✓ 训练数据已生成"
else
    echo "✓ 训练数据已存在"
fi

# ── 6. 验证环境 ──
echo ""
echo "[6/6] 验证环境..."
python3 -c "
import torch, vllm, verl, transformers, peft, aiohttp
print(f'PyTorch:      {torch.__version__}  CUDA: {torch.version.cuda}')
print(f'GPU:          {torch.cuda.get_device_name(0)}')
print(f'GPU Memory:   {torch.cuda.mem_get_info()[1]/1e9:.0f} GB')
print(f'vLLM:         {vllm.__version__}')
print(f'veRL:         {verl.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'peft:         {peft.__version__}')
"

echo ""
echo "══════════════════════════════════════════════"
echo "  ✓ 初始化完成!"
echo ""
echo "  启动训练:"
echo "    bash ${REPO_DIR}/rl/scripts/start_training.sh"
echo ""
echo "  或手动启动:"
echo "    tmux new -s train"
echo "    cd ${REPO_DIR}"
echo "    export OPENCLAW_HOST=${ECS_HOST}"
echo "    export OPENCLAW_SSH_KEY=/root/.ssh/id_ed25519"
echo "    bash rl/train/run_reinforce_lora.sh 2>&1 | tee /workspace/train.log"
echo "══════════════════════════════════════════════"
