#!/usr/bin/env bash
# 释放磁盘：删除 checkpoint 根目录下多余的 global_step_*（FSDP actor 分片数 GB，最容易撑满盘）。
#
# ── RunPod 上怎么用（在 Pod 的 Web Terminal / SSH 里执行，不是本机）──
#   cd /workspace/pinchbench-skill   # 与 setup_new_pod.sh 的 REPO_DIR 一致；若你 clone 在别处请改路径
#   df -h /workspace
#   DRY_RUN=1 bash rl/scripts/prune_old_ckpts.sh    # 先看会删谁
#   bash rl/scripts/prune_old_ckpts.sh              # 真正删除
# 若仓库不在默认路径：
#   CKPT_ROOT=/workspace/你的路径/rl/checkpoints/reinforce_lora bash rl/scripts/prune_old_ckpts.sh
#
# 默认策略：
#   - 若存在 best_ckpt_state.json（训练开了 PINCHBENCH_BEST_CKPT=1）→ 只保留该最佳步
#   - 否则 → 按步数保留最近 KEEP_LATEST_N 个目录（默认 1）
#
# 其它：
#   KEEP_LATEST_N=2 bash rl/scripts/prune_old_ckpts.sh   # 无 best 文件时保留最近 2 个
# 可选：清 TensorBoard（省一点，通常远不如删 global_step_*）
#   rm -rf /workspace/pinchbench-skill/rl/checkpoints/reinforce_lora/tensorboard/*

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/rl/checkpoints/reinforce_lora}"
KEEP_LATEST_N="${KEEP_LATEST_N:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -d "${CKPT_ROOT}" ]; then
  echo "目录不存在: ${CKPT_ROOT}"
  exit 1
fi

echo "=== 各 global_step 占用（du -sh）==="
find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' -print0 2>/dev/null | while IFS= read -r -d '' d; do
  du -sh "$d" 2>/dev/null || true
done | sort -h || true

ALL_STEPS=()
while IFS= read -r s; do
  [ -n "${s}" ] && ALL_STEPS+=("${s}")
done < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sed 's/.*global_step_//' | sort -n)

if [ "${#ALL_STEPS[@]}" -eq 0 ]; then
  echo "未发现 global_step_* 子目录，无需清理。"
  exit 0
fi

KEEP=()
STATE="${CKPT_ROOT}/best_ckpt_state.json"
if [ -f "${STATE}" ]; then
  BEST_STEP="$(python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); print(int(d['best_step']))" "${STATE}" 2>/dev/null)" || BEST_STEP=""
  if [ -n "${BEST_STEP}" ] && [ -d "${CKPT_ROOT}/global_step_${BEST_STEP}" ]; then
    KEEP=("${BEST_STEP}")
    echo "将保留 best_ckpt_state.json 中的步数: ${BEST_STEP}"
  else
    echo "WARN: best_ckpt_state.json 存在但 global_step_${BEST_STEP:-?} 缺失，改用最近 KEEP_LATEST_N=${KEEP_LATEST_N}"
  fi
fi

if [ "${#KEEP[@]}" -eq 0 ]; then
  # 取步数最大的 N 个
  count="${#ALL_STEPS[@]}"
  start=$((count - KEEP_LATEST_N))
  if [ "${start}" -lt 0 ]; then start=0; fi
  for ((i = start; i < count; i++)); do
    KEEP+=("${ALL_STEPS[i]}")
  done
  echo "将保留最近 ${KEEP_LATEST_N} 个步: ${KEEP[*]}"
fi

rm_one() {
  local step="$1"
  local path="${CKPT_ROOT}/global_step_${step}"
  if [ ! -d "${path}" ]; then
    return 0
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY_RUN] 将删除: ${path}"
  else
    echo "删除: ${path}"
    rm -rf "${path}"
  fi
}

for s in "${ALL_STEPS[@]}"; do
  keep_it=0
  for k in "${KEEP[@]}"; do
    if [ "$s" = "$k" ]; then keep_it=1; break; fi
  done
  if [ "${keep_it}" -eq 0 ]; then
    rm_one "$s"
  fi
done

if [ "${DRY_RUN}" != "1" ] && [ -f "${CKPT_ROOT}/latest_checkpointed_iteration.txt" ]; then
  # 与保留目录对齐，避免 veRL 指向已删路径
  LAST="${KEEP[-1]}"
  echo "${LAST}" > "${CKPT_ROOT}/latest_checkpointed_iteration.txt"
  echo "已写入 latest_checkpointed_iteration.txt -> ${LAST}"
fi

echo "完成。若仍满盘：可删 ${CKPT_ROOT}/tensorboard 下旧 run，或把 HuggingFace/torch 缓存移到大盘。"
