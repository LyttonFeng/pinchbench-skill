#!/usr/bin/env bash
# SFT LoRA fine-tuning on teacher exec轨迹 to activate exec tool usage.
# Base: DPO v2 merged model (dpo_merged_qwen31_task18_v2)
#
# Usage (on RunPod):
#   cd /workspace/pinchbench-skill
#   bash rl/train/run_sft_exec.sh
set -euo pipefail

MODEL_NAME="/workspace/pinchbench-skill/rl/checkpoints/dpo_merged_qwen31_task18_v2"
DATA_PATH="/workspace/pinchbench-skill/rl/data/generated/task_18_spreadsheet_summary_runtime/sft_exec_train.jsonl"
OUTPUT_DIR="/workspace/pinchbench-skill/rl/checkpoints/sft_exec_qwen31_task18"

mkdir -p "$OUTPUT_DIR"

python3 rl/train/train_sft_lora_fixed.py \
  --model-name "$MODEL_NAME" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-epochs 2 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 5e-6 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --max-length 8192 \
  --bf16 \
  2>&1 | tee "$OUTPUT_DIR/train.log"
