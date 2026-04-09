#!/bin/bash
set -euo pipefail
cd /workspace/pinchbench-skill

export PYTHONPATH="/workspace/pinchbench-skill:${PYTHONPATH:-}"
export OPENCLAW_HOST="8.163.82.224"
export OPENCLAW_USER="root"
export OPENCLAW_SSH_KEY="/root/.ssh/id_ed25519"
export PINCHBENCH_DIR="/workspace/pinchbench-skill"
export REWARD_MODE="baseline"
export PRM_VLLM_BASE_URL="http://localhost:8000/v1"
export PRM_MODEL="Qwen3-4B"
export PRM_API_KEY="dummy"
export HF_HOME="/workspace/hf_cache"
export PYTHONUNBUFFERED=1
export AGENT_TIMEOUT=180
export MAX_TURNS=10

echo "=== Starting training at $(date) ==="
bash rl/train/run_reinforce_lora.sh 2>&1 | tee /workspace/train9.log
