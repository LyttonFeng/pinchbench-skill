#!/usr/bin/env bash
# 阿里云 ECS 环境初始化脚本
# 在 ECS (Ubuntu 22.04) 上安装 OpenClaw + PinchBench 运行环境
#
# 用法（从本地执行）：
#   ssh root@8.163.82.224 'bash -s' < rl/scripts/setup_ecs.sh
#
# 或者 SSH 到 ECS 后直接运行：
#   bash setup_ecs.sh

set -euo pipefail

echo "=========================================="
echo "  阿里云 ECS 环境初始化"
echo "  目标: OpenClaw + PinchBench 运行环境"
echo "=========================================="

# ── 1. 系统更新 ──
echo "[1/7] 系统更新..."
apt-get update -y
apt-get install -y \
    curl wget git build-essential \
    python3 python3-pip python3-venv \
    jq unzip

# ── 2. 安装 Node.js 20.x ──
echo "[2/7] 安装 Node.js 20.x..."
if ! command -v node &>/dev/null || [[ "$(node -v)" != v20* ]]; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
echo "Node.js: $(node -v)"
echo "npm: $(npm -v)"

# ── 3. 安装 OpenClaw ──
echo "[3/7] 安装 OpenClaw..."
npm install -g @anthropic/openclaw 2>/dev/null || npm install -g openclaw
echo "OpenClaw: $(openclaw --version 2>/dev/null || echo 'installed')"

# ── 4. 安装 uv (Python 包管理) ──
echo "[4/7] 安装 uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ── 5. 安装 Python 依赖 ──
echo "[5/7] 安装 Python 依赖..."
pip3 install --upgrade pip
pip3 install pyyaml pypdf pdfplumber pandas pyarrow

# ── 6. 克隆 pinchbench-skill ──
echo "[6/7] 克隆 pinchbench-skill..."
REPO_DIR="/workspace/pinchbench-skill"
if [ -d "$REPO_DIR" ]; then
    echo "仓库已存在: $REPO_DIR"
    cd "$REPO_DIR" && git pull || true
else
    git clone https://github.com/anthropics/pinchbench-skill.git "$REPO_DIR" 2>/dev/null \
        || echo "请手动克隆仓库到 $REPO_DIR"
fi

# ── 7. 配置 OpenClaw ──
echo "[7/7] 配置 OpenClaw..."
OPENCLAW_DIR="$HOME/.openclaw"
mkdir -p "$OPENCLAW_DIR"

# 初始化 OpenClaw (创建默认配置)
openclaw init 2>/dev/null || true

# 配置 openclaw.json: 只保留 OpenClaw 运行所需的最小配置
# DashScope judge 走 RunPod 侧 `scripts/lib_grading.py`，不需要写进 ECS 的 OpenClaw 全局配置
OPENCLAW_CONFIG="$OPENCLAW_DIR/openclaw.json"
if [ -f "$OPENCLAW_CONFIG" ]; then
    echo "OpenClaw 配置已存在: $OPENCLAW_CONFIG"
    echo "保留现有配置；如其中仍包含 dashscope provider，建议手动清理掉"
else
    cat > "$OPENCLAW_CONFIG" << 'EOJSON'
{
  "gateway": {
    "port": 18789,
    "bind": "loopback"
  }
}
EOJSON
    echo "创建了默认 OpenClaw 配置"
fi

# 创建 workspace 目录
mkdir -p /tmp/pinchbench

echo ""
echo "=========================================="
echo "  初始化完成!"
echo ""
echo "  Node.js: $(node -v)"
echo "  npm: $(npm -v)"
echo "  Python: $(python3 --version)"
echo "  OpenClaw: $(openclaw --version 2>/dev/null || echo 'ok')"
echo ""
echo "  下一步:"
echo "  1. 确认 RunPod 的 SSH 公钥已添加到 ~/.ssh/authorized_keys"
echo "  2. 测试 OpenClaw:"
echo "     openclaw agent --message 'hello' --local"
echo "=========================================="
