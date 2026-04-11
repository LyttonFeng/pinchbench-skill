#!/usr/bin/env bash
# Baseline RL8 benchmark for Qwen/Qwen3-4B with per-task runs=3.
#
# Usage:
#   bash scripts/run_bench_rl8_runs3_baseline.sh
#   BASE_URL=http://127.0.0.1:18010/v1 bash scripts/run_bench_rl8_runs3_baseline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/scripts"

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

MODEL="${MODEL:-Qwen/Qwen3-4B}"
BASE_URL="${BASE_URL:-http://127.0.0.1:18010/v1}"
API_KEY="${API_KEY:-dummy}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen-plus}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/results/compare/baseline_qwen3_rl8_runs3_${TS}}"

mkdir -p "${OUTDIR}"

echo "RL8_SUITE=${RL8_SUITE}"
echo "MODEL=${MODEL} BASE_URL=${BASE_URL}"
echo "OUTDIR=${OUTDIR}"

python3 -u benchmark.py \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --suite "${RL8_SUITE}" \
  --runs 3 \
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
