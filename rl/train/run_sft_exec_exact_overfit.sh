#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen3-1.7B"
DATA_PATH="/workspace/pinchbench-skill/rl/data/generated/task_18_spreadsheet_summary_runtime/sft_exact_prompt_overfit_train.jsonl"
OUTPUT_DIR="/workspace/pinchbench-skill/rl/checkpoints/sft_exact_prompt_overfit_qwen31_task18"

mkdir -p "$OUTPUT_DIR"

python3 rl/train/train_sft_lora_fixed.py \
  --model-name "$MODEL_NAME" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-epochs 20 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 2e-5 \
  --lora-rank 128 \
  --lora-alpha 256 \
  --max-length 8192 \
  --bf16 \
  2>&1 | tee "$OUTPUT_DIR/train.log"
