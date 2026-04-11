#!/usr/bin/env bash
# PinchBench RL8 — LoRA global_step_25（API 模型名默认 pinchbench-lora，端口默认 8011）
#
# 前置：先起带 LoRA 的 vLLM，例如：
#   export LORA_ADAPTER_PATH=.../global_step_25/actor/lora_adapter
#   export VLLM_LORA_NAME=pinchbench-lora
#   export VLLM_PORT=8011
#   bash rl/scripts/start_vllm.sh Qwen/Qwen3-4B
#
# 用法：
#   bash scripts/run_bench_rl8_lora.sh
#   SAVE_RL8_COMPARE=1 bash scripts/run_bench_rl8_lora.sh   # 写入 results/compare/lora_rl8_step25_qwen3_4b.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL="${MODEL:-pinchbench-lora}"
export BASE_URL="${BASE_URL:-http://127.0.0.1:18010/v1}"
export API_KEY="${API_KEY:-dummy}"
export RL8_COMPARE_PREFIX="${RL8_COMPARE_PREFIX:-lora_rl8_step25_qwen3_4b}"

wait_for_models() {
  local url="$1"
  local label="$2"
  echo "[wait] ${label} ${url}/models ..."
  for _ in $(seq 1 60); do
    if curl -sf "${url}/models" >/dev/null 2>&1; then
      echo "[wait] ${label} ready."
      return 0
    fi
    sleep 2
  done
  echo "ERROR: ${label} was not ready within 120s" >&2
  return 1
}

if ! curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: ${BASE_URL}/models is not reachable."
  echo "       Start or fix the SSH tunnel first, then rerun."
  echo "       Example:"
  echo "         ssh -fN -o ServerAliveInterval=30 -L 127.0.0.1:18010:127.0.0.1:8010 root@213.173.105.9 -p 40054 -i ~/.ssh/id_ed25519"
  if [[ "${AUTO_TUNNEL:-0}" == "1" ]]; then
    host="${LORA_TUNNEL_HOST:-213.173.105.9}"
    port="${LORA_TUNNEL_PORT:-40054}"
    user="${LORA_TUNNEL_USER:-root}"
    key="${LORA_TUNNEL_KEY:-${HOME}/.ssh/id_ed25519}"
    remote_port="${LORA_REMOTE_PORT:-8010}"
    local_port="${LORA_LOCAL_PORT:-18010}"
    echo "[auto] AUTO_TUNNEL=1 set, starting SSH tunnel ${local_port}:127.0.0.1:${remote_port} -> ${user}@${host}:${port}"
    ssh -fN -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
      -L "127.0.0.1:${local_port}:127.0.0.1:${remote_port}" \
      -i "${key}" "${user}@${host}" -p "${port}" &
    wait_for_models "${BASE_URL}" "loRA-vLLM tunnel"
  else
    exit 1
  fi
fi

exec bash "${REPO_ROOT}/scripts/run_bench_rl8.sh" "$@"
