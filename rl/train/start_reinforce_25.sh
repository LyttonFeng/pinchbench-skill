#!/usr/bin/env bash
# 短跑：25 次 PPO 迭代（veRL global_steps / trainer.total_training_steps）。
# 依赖与 run_reinforce_lora.sh 相同；启动前仍会做 train/infer parity、DashScope qwen-plus、ECS SSH OpenClaw 预检。
#
# 必填（在训练机上）:
#   export OPENCLAW_HOST=<ECS 公网 IP>
#   export DASHSCOPE_API_KEY=sk-...   # 或写入 ~/.pinchbench_env
#
# 用法:
#   bash rl/train/start_reinforce_25.sh
#
# 续跑同一 OUTPUT_DIR 时: export TRAINER_RESUME_MODE=auto
# 全新任务（推荐）: 默认 TRAINER_RESUME_MODE=disable；必要时手动清理 rl/checkpoints/reinforce_lora/global_step_*

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
export TOTAL_TRAINING_STEPS=25
# save_freq 在 PINCHBENCH_BEST_CKPT=1 时默认等于 test_freq；5/10/15/20/25 各 val+存盘
export TEST_FREQ="${TEST_FREQ:-5}"
export PINCHBENCH_BEST_CKPT="${PINCHBENCH_BEST_CKPT:-1}"
export PINCHBENCH_KEEP_LATEST_CKPT="${PINCHBENCH_KEEP_LATEST_CKPT:-1}"

exec bash "${REPO_ROOT}/rl/train/run_reinforce_lora.sh"
