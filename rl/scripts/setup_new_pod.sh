#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  RunPod 新 Pod 一键初始化脚本
#
#  推荐镜像: runpod/pytorch:2.10.0-py3.12-cuda12.8.0-devel-ubuntu22.04
#             (自带 PyTorch 2.10 + Python 3.12 + CUDA 12.8)
#  GPU 推荐: NVIDIA L40S (48GB) 或 A100 (80GB)
#
#  用法:
#    1. 新建 Pod，SSH 进去
#    2. 运行: bash setup_new_pod.sh
#    3. 按提示启动训练
#
#  ── 踩坑记录 ──
#
#  1. flash-attn 绝对不能从源码编译 (pip install flash-attn)
#     → 会耗尽 CPU/RAM 导致 Pod 冻结。必须用预编译 wheel。
#     → wheel 地址见下方 step 2c
#
#  2. 安装顺序: PyTorch → vLLM → flash-attn → veRL
#     → vLLM 会拉它自己的 torch 版本，必须先装好 PyTorch
#     → flash-attn wheel 必须匹配 torch 版本 + CUDA 版本 + Python 版本
#     → veRL 最后装，避免它改 torch/vllm 依赖
#
#  3. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#     → vLLM 0.19 的 CuMemAllocator 会 assert 报错，绝对不能设
#
#  4. gpu_memory_utilization 不能太高
#     → 0.5 → OOM (backward pass 需要 ~10GB)
#     → 0.4 → 还是 OOM
#     → 0.35 → 配合 optimizer_offload=True 和 batch_size=1 可以跑
#     → A100 80GB 可以用更高的值
#
#  5. ECS 4核8G 资源有限
#     → 并发 OpenClaw 不要超过 1-2 个
#     → openclaw agents add 需要 flock 防并发写
#     → agent_timeout 需要 300s+
#
#  6. OpenClaw SSE 流式 tool_calls 格式
#     → 必须单 chunk 发送每个 tool_call（含 id + name + arguments）
#     → 不能分成 name chunk + args chunk
#     → 不能在 tool_calls 响应里混 content chunk
#
#  7. Qwen3 的 <tool_call>/<think> 是 special tokens
#     → skip_special_tokens=True 会移除 tag 但保留 JSON body
#     → _parse_tool_calls 必须用 skip_special_tokens=False 的文本
#
#  8. OpenClaw system prompt 约 14000 tokens
#     → max_prompt_length 必须 ≥ 16384，否则 prompt 被截断丢失 tool 定义
#
#  9. 磁盘满 → checkpoint 保存失败
#     → PyTorch 报错: unexpected pos ... inline_container.cc（实为写盘失败）
#     → df -h /workspace；HF_HOME、docker 层、旧 global_step_* 都会占空间
#     → 在 Pod 里清旧 ckpt: cd /workspace/pinchbench-skill && DRY_RUN=1 bash rl/scripts/prune_old_ckpts.sh
#       再 bash rl/scripts/prune_old_ckpts.sh（有 best_ckpt_state.json 则只留最佳步）
#     → run_reinforce_lora.sh: 默认 PINCHBENCH_BEST_CKPT=1 会按 val 只留最佳；否则可调 save_freq / max_actor_ckpt
#
#  ECS：公网 IP 会变，运行前在 shell 里 export（勿写死在仓库）：
#    export ECS_HOST=<阿里云控制台显示的公网 IP>
#    User/Port 默认 root / 22，可用 ECS_USER / ECS_PORT 覆盖
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

if [ -z "${ECS_HOST:-}" ]; then
    echo "ERROR: 请先设置 ECS_HOST 为当前 ECS 公网 IP（云控制台查看，非固定）:"
    echo "  export ECS_HOST=<...>"
    exit 1
fi
ECS_USER="${ECS_USER:-root}"
ECS_PORT="${ECS_PORT:-22}"
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

# 检测已有环境
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
echo "  当前 PyTorch: ${TORCH_VER}"

# 2a. PyTorch 2.10 (cu128)
if [[ "$TORCH_VER" != 2.10* ]]; then
    echo "  安装 PyTorch 2.10 (cu128)..."
    pip install --break-system-packages \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128
else
    echo "  ✓ PyTorch 已是 2.10"
fi

# 2b. vLLM 0.19.0
VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "none")
if [[ "$VLLM_VER" != "0.19.0" ]]; then
    echo "  安装 vllm 0.19.0..."
    pip install --break-system-packages vllm==0.19.0
else
    echo "  ✓ vLLM 已是 0.19.0"
fi

# 2c. flash-attn: 预编译 wheel，绝对不从源码编译！！！
#     匹配: Python 3.12 + PyTorch 2.10 + CUDA 12.x
#     来源: https://github.com/lesj0610/flash-attention/releases
FA_VER=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "none")
if [[ "$FA_VER" == "none" ]]; then
    PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    echo "  安装 flash-attn (预编译 wheel, Python ${PY_VER})..."
    if [[ "$PY_VER" == "cp312" ]]; then
        pip install --break-system-packages \
            "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    elif [[ "$PY_VER" == "cp311" ]]; then
        pip install --break-system-packages \
            "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp311/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    else
        echo "  ⚠ 无匹配的 flash-attn wheel (Python ${PY_VER})"
        echo "  尝试 pip install flash-attn --no-build-isolation ..."
        pip install --break-system-packages flash-attn --no-build-isolation || echo "  ⚠ flash-attn 安装失败，继续"
    fi
else
    echo "  ✓ flash-attn 已安装: ${FA_VER}"
fi

# 2d. verl + 其他依赖
VERL_VER=$(python3 -c "import verl; print(verl.__version__)" 2>/dev/null || echo "none")
if [[ "$VERL_VER" == "none" ]]; then
    echo "  安装 verl 及其他依赖..."
    pip install --break-system-packages \
        verl==0.7.1 \
        peft \
        accelerate \
        "datasets>=3.0"
else
    echo "  ✓ veRL 已安装: ${VERL_VER}"
fi

# 2e. TensorBoard（veRL logger=tensorboard 时必需；否则无 events 文件）
if ! python3 -c "import tensorboard" 2>/dev/null; then
    echo "  安装 tensorboard..."
    pip install --break-system-packages tensorboard
else
    echo "  ✓ tensorboard 已安装"
fi

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
try:
    import flash_attn
    print(f'flash-attn:   {flash_attn.__version__}')
except ImportError:
    print('flash-attn:   NOT INSTALLED')
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
