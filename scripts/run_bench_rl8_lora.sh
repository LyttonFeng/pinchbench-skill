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
export BASE_URL="${BASE_URL:-http://127.0.0.1:8011/v1}"
export API_KEY="${API_KEY:-dummy}"
export RL8_COMPARE_PREFIX="${RL8_COMPARE_PREFIX:-lora_rl8_step25_qwen3_4b}"

exec bash "${REPO_ROOT}/scripts/run_bench_rl8.sh" "$@"
