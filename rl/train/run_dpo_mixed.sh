#!/bin/bash
# Focused + Full mixed DPO training for task_18_spreadsheet_summary exec activation
# Base: Qwen3-1.7B (not DPO v2)
# Data: dpo_pairs_mixed_train.jsonl (39 pairs: 19 focused + 20 full)
# LoRA rank: 128 (vs v2's 32)
# Epochs: 5 (vs v2's 3)

set -e

MODEL_NAME="Qwen/Qwen3-1.7B"
DATA_PATH="rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_mixed_train.jsonl"
OUTPUT_DIR="rl/checkpoints/dpo_mixed_lora"

python3 rl/train/train_dpo_lora.py \
    --model-name "$MODEL_NAME" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs 5 \
    --batch-size 1 \
    --grad-accum 4 \
    --lr 5e-6 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --max-length 4096 \
    --bf16
