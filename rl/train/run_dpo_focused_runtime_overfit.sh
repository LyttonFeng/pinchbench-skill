#!/usr/bin/env bash
# Tiny repeated focused-DPO overfit sanity for task_18 under runtime parity.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
DATA_PATH="${DATA_PATH:-/workspace/pinchbench-skill/rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_focused_runtime_overfit_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/pinchbench-skill/rl/checkpoints/dpo_focused_runtime_overfit_qwen31_task18}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-7800}"
PROMPT_TRUNCATION_MODE="${PROMPT_TRUNCATION_MODE:-keep_end}"

cd /workspace/pinchbench-skill

python3 rl/train/train_dpo_lora_fixed.py \
  --model-name "$MODEL_NAME" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-epochs 12 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 2e-5 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --max-length "$MAX_LENGTH" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --prompt-truncation-mode "$PROMPT_TRUNCATION_MODE" \
  --beta 0.1 \
  --bf16
