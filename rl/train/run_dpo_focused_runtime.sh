#!/usr/bin/env bash
# Focused DPO with train/infer parity for task_18_spreadsheet_summary

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
DATA_PATH="${DATA_PATH:-/workspace/pinchbench-skill/rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_focused_runtime_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/pinchbench-skill/rl/checkpoints/dpo_focused_runtime_qwen31_task18}"

cd /workspace/pinchbench-skill

python3 rl/train/train_dpo_lora_fixed.py \
  --model-name "$MODEL_NAME" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-epochs 3 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 1e-5 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --max-length 8192 \
  --beta 0.1 \
  --bf16
