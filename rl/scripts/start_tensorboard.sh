#!/usr/bin/env bash
# 在 RunPod / 本机查看训练曲线（需先跑过带 tensorboard 的训练）
#
# 用法:
#   bash rl/scripts/start_tensorboard.sh
#   bash rl/scripts/start_tensorboard.sh /path/to/tensorboard_log
#
# 默认日志目录: rl/checkpoints/reinforce_lora/tensorboard
# 浏览器打开: http://<host>:6006

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOGDIR="${1:-${REPO_ROOT}/rl/checkpoints/reinforce_lora/tensorboard}"

if [[ ! -d "${LOGDIR}" ]]; then
  echo "目录不存在: ${LOGDIR}"
  echo "请先训练（run_reinforce_lora.sh 已启用 tensorboard），或传入 TENSORBOARD_DIR"
  exit 1
fi

echo "TensorBoard logdir: ${LOGDIR}"
echo "启动: tensorboard --logdir=${LOGDIR} --bind_all --port=${TENSORBOARD_PORT:-6006}"
exec tensorboard --logdir="${LOGDIR}" --bind_all --port="${TENSORBOARD_PORT:-6006}"
