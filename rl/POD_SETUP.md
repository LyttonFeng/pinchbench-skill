# RunPod Environment Setup for DPO/RL Training

## Pod Specs
- GPU: NVIDIA L40S (46GB VRAM)
- OS: Ubuntu 24.04.3 LTS
- CUDA: 12.8
- Python: 3.12

## Installed Packages

```bash
pip install --upgrade transformers trl --break-system-packages
```

Key versions (after upgrade):
- transformers: latest (supports MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
- trl: latest (DPO trainer compatible)
- torch: pre-installed
- peft: pre-installed
- vllm: pre-installed

## Directory Structure

```
~/pinchbench-skill/
├── rl/
│   ├── checkpoints/
│   │   ├── dpo_mixed_lora/          # DPO LoRA adapter (rank 32)
│   │   └── dpo_mixed_merged/        # Merged full model (3.3GB)
│   ├── data/generated/task_18_spreadsheet_summary_runtime/
│   │   ├── dpo_pairs_clean_train.jsonl      # 20 full pairs
│   │   ├── dpo_pairs_focused_train.jsonl    # 19 focused pairs
│   │   └── dpo_pairs_mixed_train.jsonl      # 39 mixed pairs
│   ├── train/
│   │   ├── train_dpo_lora.py
│   │   └── run_dpo_mixed.sh
│   └── verl_lora_only_ckpt_patch.py
└── vllm_new.log
```

## Training Configuration

### DPO Mixed Training (run_dpo_mixed.sh)
```bash
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
```

### Key Training Modifications (train_dpo_lora.py)
1. **ref_model=None**: Share reference with model to save memory
2. **gradient_checkpointing=True**: Reduce activation memory
3. **max_length=4096**: Reduced from 8192 to fit in 46GB VRAM

### Merge LoRA to Full Model
```bash
cd ~/pinchbench-skill
python3 -c '
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen3-1.7B"
lora_path = "rl/checkpoints/dpo_mixed_lora"
output_path = "rl/checkpoints/dpo_mixed_merged"

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="cpu", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
'
```

### vLLM Inference
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /root/pinchbench-skill/rl/checkpoints/dpo_mixed_merged \
  --served-model-name Qwen3-1.7B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  > vllm_new.log 2>&1 &
```

## Known Issues & Solutions

### Issue 1: DPO OOM with 46GB VRAM
**Problem**: DPO needs model + ref_model + activations, exceeds 46GB

**Solutions applied**:
1. Set `ref_model=None` (share reference)
2. Enable `gradient_checkpointing=True`
3. Reduce `max_length` from 8192 to 4096
4. Keep `batch_size=1`

### Issue 2: transformers/trl version incompatibility
**Error**: `cannot import name 'MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES'`

**Solution**:
```bash
pip install --upgrade transformers trl --break-system-packages
```

### Issue 3: vLLM doesn't recognize local model path
**Error**: `Repo id must be in the form 'repo_name' or 'namespace/repo_name'`

**Solution**: Add `--trust-remote-code` flag

## Training Results

### DPO Mixed (39 pairs, 5 epochs, rank 32)
- Training time: ~7 minutes
- Loss: 0.06851 → 3e-15 (converged)
- Rewards margin: 8.41 → 34.34
- Accuracies: 100%

### Benchmark Results
- **task_18_spreadsheet_summary**: 5.8% (0.2/3.0)
- **Issue**: Model did NOT activate exec, only called read
- **Root cause**: Unknown - data has exec+openpyxl in chosen

## Data Verification

Checked `dpo_pairs_mixed_train.jsonl`:
- ✅ Chosen has `exec` with `openpyxl` commands
- ✅ Rejected has `read` (wrong tool)
- ✅ 39 pairs total (19 focused + 20 full)
- ✅ Data structure correct

## Next Steps for New Pod

1. Install packages: `pip install --upgrade transformers trl --break-system-packages`
2. Sync data: `rsync -avz rl/data/ root@NEW_POD:~/pinchbench-skill/rl/data/`
3. Sync scripts: `rsync -avz rl/train/ root@NEW_POD:~/pinchbench-skill/rl/train/`
4. Consider larger GPU (A100 80GB) to:
   - Use full `max_length=8192`
   - Train with separate ref_model
   - Increase LoRA rank to 128
