#!/usr/bin/env bash
# RunPod：拉最新代码 → 校验 tasks/ 与 rsync → 启动 REINFORCE LoRA 训练
#
# 用法（在 Pod 里，仓库根目录或任意目录）：
#   bash rl/scripts/runpod_pull_and_train.sh
#   bash rl/scripts/runpod_pull_and_train.sh   # 透传给 run_reinforce_lora 的环境可先 export 或写 ~/.pinchbench_env
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "${HOME}/.pinchbench_env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${HOME}/.pinchbench_env"
  set +a
fi

echo "[runpod_pull_and_train] repo: ${REPO_ROOT}"
git pull --ff-only

if [ ! -d "${REPO_ROOT}/tasks" ] || [ ! -f "${REPO_ROOT}/tasks/task_00_sanity.md" ]; then
  echo "ERROR: ${REPO_ROOT}/tasks 不完整，PINCHBENCH 任务无法加载"
  exit 1
fi

export PINCHBENCH_DIR="${REPO_ROOT}"
echo "[runpod_pull_and_train] PINCHBENCH_DIR=${PINCHBENCH_DIR}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: 需要 rsync（远程 OpenClaw 工作区同步到 ECS）。请: apt-get update && apt-get install -y rsync"
  exit 1
fi

_host="${OPENCLAW_HOST:-}"
if [ -z "${_host}" ] || [ "${_host}" = "localhost" ] || [ "${_host}" = "127.0.0.1" ]; then
  echo "ERROR: 请先在环境或 ~/.pinchbench_env 里设置 OPENCLAW_HOST=<ECS 公网 IP>"
  exit 1
fi

_user="${OPENCLAW_USER:-root}"
_port="${OPENCLAW_PORT:-22}"
_key="${OPENCLAW_SSH_KEY:-${HOME}/.ssh/id_ed25519}"
echo "[runpod_pull_and_train] preflight: ssh+rsync -> ${_user}@${_host}:${_port}"

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
  -i "${_key}" -p "${_port}" \
  "${_user}@${_host}" "mkdir -p /tmp/pinchbench && echo ok" >/dev/null

TMPDIR_RSYNC="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_RSYNC}"' EXIT
echo "pinchbench_rsync_test" > "${TMPDIR_RSYNC}/.pinchbench_rsync_test"
rsync -az --timeout=30 \
  -e "ssh -o StrictHostKeyChecking=no -i ${_key} -p ${_port}" \
  "${TMPDIR_RSYNC}/" "${_user}@${_host}:/tmp/pinchbench/.pinchbench_rsync_test_dir/"
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
  -i "${_key}" -p "${_port}" \
  "${_user}@${_host}" "rm -rf /tmp/pinchbench/.pinchbench_rsync_test_dir" >/dev/null
echo "[runpod_pull_and_train] ssh + rsync OK"

exec bash "${REPO_ROOT}/rl/train/run_reinforce_lora.sh" "$@"
