#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  在 tmux 中启动 RL 训练
#
#  用法: bash rl/scripts/start_training.sh
#
#  ECS 信息:
#    IP:   8.163.82.224
#    User: root
#    Port: 22
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

REPO_DIR="/workspace/pinchbench-skill"
LOG_FILE="/workspace/train_$(date +%Y%m%d_%H%M).log"
TMUX_SESSION="train"

# ── 环境变量 ──
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export OPENCLAW_HOST="8.163.82.224"
export OPENCLAW_USER="root"
export OPENCLAW_SSH_KEY="/root/.ssh/id_ed25519"
export OPENCLAW_PORT="22"
export PINCHBENCH_DIR="${REPO_DIR}"
export REWARD_MODE="self-judge"
export PRM_VLLM_BASE_URL="http://localhost:9090/v1"
export PRM_MODEL="Qwen3-4B"
export PRM_API_KEY="dummy"
export HF_HOME="/workspace/hf_cache"
export PYTHONUNBUFFERED=1
export AGENT_TIMEOUT=120
export MAX_TURNS=5

cd "${REPO_DIR}"

echo "══════════════════════════════════════"
echo "  启动 RL 训练"
echo "  日志: ${LOG_FILE}"
echo "  tmux session: ${TMUX_SESSION}"
echo "══════════════════════════════════════"

# 检查是否已有训练在跑
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "WARNING: tmux session '${TMUX_SESSION}' 已存在"
    echo "  查看: tmux attach -t ${TMUX_SESSION}"
    echo "  杀掉: tmux kill-session -t ${TMUX_SESSION}"
    read -p "是否杀掉旧 session 并重启? [y/N] " -r
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        tmux kill-session -t "${TMUX_SESSION}"
    else
        exit 1
    fi
fi

# 先验证 SSH 连接
echo "验证 ECS SSH 连接..."
if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -i "${OPENCLAW_SSH_KEY}" "${OPENCLAW_USER}@${OPENCLAW_HOST}" -p "${OPENCLAW_PORT}" \
    "echo 'ECS SSH OK'" 2>/dev/null; then
    echo "✗ 无法连接 ECS，请检查 SSH key 和网络"
    exit 1
fi
echo "✓ ECS 连接正常"

# 验证 GPU
echo "验证 GPU..."
python3 -c "
import torch
free, total = torch.cuda.mem_get_info()
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total')
if free < 30e9:
    print(f'WARNING: 可用内存 {free/1e9:.1f}GB 可能不足')
"

# 验证训练数据
if [ ! -f rl/data/prompts/train.parquet ]; then
    echo "生成训练数据..."
    python3 rl/train/prepare_prompts.py --tasks-dir tasks/ --output-dir rl/data/prompts/
fi

# 在 tmux 中启动训练
echo ""
echo "在 tmux 中启动训练..."
tmux new-session -d -s "${TMUX_SESSION}" bash -c "
    cd ${REPO_DIR}
    export PYTHONPATH='${REPO_DIR}:${PYTHONPATH:-}'
    export OPENCLAW_HOST='${OPENCLAW_HOST}'
    export OPENCLAW_USER='${OPENCLAW_USER}'
    export OPENCLAW_SSH_KEY='${OPENCLAW_SSH_KEY}'
    export OPENCLAW_PORT='${OPENCLAW_PORT}'
    export PINCHBENCH_DIR='${REPO_DIR}'
    export REWARD_MODE='${REWARD_MODE}'
    export PRM_VLLM_BASE_URL='${PRM_VLLM_BASE_URL}'
    export PRM_MODEL='${PRM_MODEL}'
    export PRM_API_KEY='${PRM_API_KEY}'
    export HF_HOME='${HF_HOME}'
    export PYTHONUNBUFFERED=1
    export AGENT_TIMEOUT=${AGENT_TIMEOUT}
    export MAX_TURNS=${MAX_TURNS}

    echo '=== Training started at \$(date) ===' | tee ${LOG_FILE}
    bash rl/train/run_reinforce_lora.sh 2>&1 | tee -a ${LOG_FILE}
    echo '=== Training ended at \$(date) ===' | tee -a ${LOG_FILE}
    echo 'Press Enter to close...'
    read
"

echo ""
echo "══════════════════════════════════════"
echo "  ✓ 训练已在后台启动"
echo ""
echo "  查看: tmux attach -t ${TMUX_SESSION}"
echo "  日志: tail -f ${LOG_FILE}"
echo "  停止: tmux kill-session -t ${TMUX_SESSION}"
echo "══════════════════════════════════════"
