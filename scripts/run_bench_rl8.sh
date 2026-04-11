#!/usr/bin/env bash
# PinchBench — RL 训练用的 8 题套件（与 rl/train/prepare_prompts / compare_benchmark_json 对齐）
#
# 用法:
#   bash scripts/run_bench_rl8.sh
#   MODEL=Qwen/Qwen3-4B BASE_URL=http://127.0.0.1:8010/v1 bash scripts/run_bench_rl8.sh
#   SAVE_RL8_COMPARE=1 bash scripts/run_bench_rl8.sh   # 拷到 results/compare/<前缀>.json
#   RL8_COMPARE_PREFIX=lora_rl8_step25_qwen3_4b SAVE_RL8_COMPARE=1 ...  # 自定义文件名前缀（默认 baseline_rl8_qwen3_4b）
#
# 依赖: 本机 openclaw；判分需 DASHSCOPE_API_KEY（~/.pinchbench_env 或 仓库根 .env）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/scripts"

# 8 题 task_id（勿改顺序则便于和旧表对照；与 tasks/*.md 中 id 一致）
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

if ! curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: ${BASE_URL}/models is not reachable."
  echo "       Start or fix the SSH tunnel first, then rerun."
  echo "       Example:"
  echo "         ssh -fN -o ServerAliveInterval=30 -L 127.0.0.1:18010:127.0.0.1:8010 root@213.173.105.9 -p 40054 -i ~/.ssh/id_ed25519"
  exit 1
fi

echo "RL8_SUITE=${RL8_SUITE}"
echo "MODEL=${MODEL} BASE_URL=${BASE_URL}"

python3 -u benchmark.py \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --suite "${RL8_SUITE}" \
  --output-dir ../results \
  --no-fail-fast \
  "$@"

if [[ "${SAVE_RL8_COMPARE:-0}" == "1" ]]; then
  CMP="${REPO_ROOT}/results/compare"
  mkdir -p "${CMP}"
  PREFIX="${RL8_COMPARE_PREFIX:-baseline_rl8_qwen3_4b}"
  # 最新一次 benchmark 输出：{run_id}_{model_slug}.json（run_id 以数字开头）
  LATEST="$(ls -t "${REPO_ROOT}/results/"[0-9]*_*.json 2>/dev/null | grep -v -- '-best\.json$' | head -1 || true)"
  if [[ -n "${LATEST}" ]]; then
    cp -a "${LATEST}" "${CMP}/${PREFIX}.json"
    BASENAME="$(basename "${LATEST}" .json)"
    RID="${BASENAME%%_*}"
    if [[ -d "${REPO_ROOT}/results/${RID}_transcripts" ]]; then
      rm -rf "${CMP}/${PREFIX}_transcripts"
      cp -a "${REPO_ROOT}/results/${RID}_transcripts" "${CMP}/${PREFIX}_transcripts"
    fi
    echo "Saved compare snapshot → ${CMP}/${PREFIX}.json (run_id ${RID})"
  else
    echo "WARN: no [0-9]*_*.json benchmark output found under results/" >&2
  fi
fi
