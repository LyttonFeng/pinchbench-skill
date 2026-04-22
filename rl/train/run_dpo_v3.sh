#!/bin/bash
# DPO Training v3 - Bug 5 Fix (只用第一个 turn，避免二进制垃圾)

set -e

echo "=========================================="
echo "DPO Training v3 - Bug 5 Fix"
echo "=========================================="

# Step 1: 清理旧 checkpoint
echo ""
echo "Step 1: 清理旧 checkpoint..."
rm -rf ~/pinchbench-skill/rl/checkpoints/dpo_pure_focused_fixed_v3
rm -rf ~/pinchbench-skill/rl/checkpoints/dpo_pure_focused_fixed_v3_merged

# Step 2: 训练 DPO LoRA
echo ""
echo "Step 2: 开始 DPO 训练..."
cd ~/pinchbench-skill

python3 rl/train/train_dpo_lora_fixed.py \
    --model-name Qwen/Qwen3-1.7B \
    --data-path rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_pure_focused_train.jsonl \
    --output-dir rl/checkpoints/dpo_pure_focused_fixed_v3 \
    --num-epochs 5 \
    --batch-size 1 \
    --grad-accum 4 \
    --lr 5e-6 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --max-length 4096 \
    --bf16

echo ""
echo "✅ 训练完成！"

# Step 3: Merge LoRA
echo ""
echo "Step 3: Merge LoRA adapter..."
python3 -c '
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", device_map="cpu", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, "rl/checkpoints/dpo_pure_focused_fixed_v3")

print("Merging...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("rl/checkpoints/dpo_pure_focused_fixed_v3_merged")
tokenizer.save_pretrained("rl/checkpoints/dpo_pure_focused_fixed_v3_merged")

print("✅ Merge 完成！")
'

# Step 4: 启动 vLLM
echo ""
echo "Step 4: 启动 vLLM server..."
pkill -9 -f vllm || true
sleep 2

python3 -m vllm.entrypoints.openai.api_server \
  --model /root/pinchbench-skill/rl/checkpoints/dpo_pure_focused_fixed_v3_merged \
  --served-model-name Qwen3-1.7B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  > vllm_dpo_v3.log 2>&1 &

echo ""
echo "等待 vLLM 启动..."
sleep 30

# Step 5: 测试 vLLM
echo ""
echo "Step 5: 测试 vLLM..."
curl -s http://localhost:8000/v1/models | jq .

echo ""
echo "=========================================="
echo "✅ 全部完成！"
echo "=========================================="
echo ""
echo "下一步（在 Mac 本地）："
echo "1. ssh -f -N -L 8000:localhost:8000 runpod-zilong"
echo "2. python3 scripts/benchmark.py \\"
echo "     --model openai/Qwen3-1.7B \\"
echo "     --base-url http://localhost:8000/v1 \\"
echo "     --suite task_18_spreadsheet_summary \\"
echo "     --runs 3 \\"
echo "     --output-dir .pinchbench_runs/dpo_pure_focused_fixed_v3"
