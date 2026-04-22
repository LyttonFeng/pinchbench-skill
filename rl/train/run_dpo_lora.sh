#!/usr/bin/env bash
# DPO LoRA fine-tuning for Qwen3-1.7B on task_18_spreadsheet_summary pairs.
# Teacher: qwen3.6-plus (score=1.0), Student/rejected: Qwen3-1.7B (score~0.07)
#
# Usage (on RunPod):
#   cd /workspace/pinchbench-skill
#   bash rl/train/run_dpo_lora.sh
set -euo pipefail

MODEL_NAME="Qwen/Qwen3-1.7B"
DATA_PATH="/workspace/pinchbench-skill/rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_clean_train.jsonl"
OUTPUT_DIR="/workspace/pinchbench-skill/rl/checkpoints/dpo_lora_qwen31_task18_v2"

mkdir -p "$OUTPUT_DIR"

python3 rl/train/train_dpo_lora.py \
  --model-name "$MODEL_NAME" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-epochs 3 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 1e-5 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --max-length 4096 \
  --beta 0.1 \
  --bf16 \
  2>&1 | tee "$OUTPUT_DIR/train.log"
