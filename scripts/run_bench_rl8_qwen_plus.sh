#!/usr/bin/env bash
# PinchBench RL8 — agent + judge 均为 DashScope qwen-plus，每题只跑 1 轮。
#
# qwen-plus 成本高、方差相对小，RL8 对照一般跑一轮即可（勿与 Qwen3-4B 的 runs=3 对齐）。
#
# Usage:
#   bash scripts/run_bench_rl8_qwen_plus.sh
#   需 ~/.pinchbench_env 或仓库 .env 中有 DASHSCOPE_API_KEY（或 export API_KEY=sk-...）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/scripts"

# 仅 8 题 RL8 套件（与 run_bench_rl8*.sh 一致），不是 25 task 全量。
RL8_SUITE="task_02_stock,task_12_skill_search,task_10_workflow,task_22_second_brain,task_16_email_triage,task_18_spreadsheet_summary,task_18_market_research,task_24_polymarket_briefing"

if [[ -f "${HOME}/.pinchbench_env" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.pinchbench_env"
fi
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env"
  set +a
fi

MODEL="${MODEL:-qwen-plus}"
BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
API_KEY="${API_KEY:-${DASHSCOPE_API_KEY:-}}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen-plus}"
RUNS="${RUNS:-1}"

if [[ -z "${API_KEY}" || "${API_KEY}" == "dummy" ]]; then
  echo "ERROR: 需要 DashScope API key。请设置 DASHSCOPE_API_KEY 或 API_KEY。" >&2
  exit 1
fi

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/results/compare/rl8_qwen_plus_runs${RUNS}_${TS}}"

mkdir -p "${OUTDIR}"

echo "RL8_SUITE=${RL8_SUITE}"
echo "MODEL=${MODEL} BASE_URL=${BASE_URL} RUNS=${RUNS} (qwen-plus: 默认 1 轮即可)"
echo "OUTDIR=${OUTDIR}"

python3 -u benchmark.py \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --suite "${RL8_SUITE}" \
  --runs "${RUNS}" \
  --output-dir ../results \
  --no-fail-fast \
  --no-upload \
  --judge "${JUDGE_MODEL}" \
  2>&1 | tee "${OUTDIR}/run.log"

LATEST="$(ls -t "${REPO_ROOT}/results/"[0-9]*_*.json 2>/dev/null | grep -v -- '-best\.json$' | head -1 || true)"
if [[ -n "${LATEST}" ]]; then
  cp -a "${LATEST}" "${OUTDIR}/result.json"
  echo "Saved result snapshot -> ${OUTDIR}/result.json"
else
  echo "WARN: no benchmark JSON found under results/" >&2
fi
