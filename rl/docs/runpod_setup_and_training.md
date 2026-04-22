# RunPod 环境配置与训练流程

## 实验总结

### 已完成的实验

| 实验 | 方法 | 数据 | 配置 | 结果 | 结论 |
|------|------|------|------|------|------|
| Baseline | - | - | Qwen3-1.7B | 2.5% | 只用 read，从不用 exec |
| DPO v2 | DPO | 16 pairs (完整轨迹) | rank 32, 5 epochs | 5.8% | 数据有 Bug 5（长度不平衡） |
| DPO v3 | DPO | 16 pairs (只第一个turn) | rank 32, 5 epochs | 2.5% | Bug 5 修复，但无法激活能力 |
| SFT v1 | SFT | 20 teacher samples | rank 16, 3 epochs | 2.5% | 数据量太少 + rank 太小 |

### 核心发现

1. **DPO 无法从零激活新能力**
   - Baseline 只有 5% 的 exec 使用率
   - DPO 无法将 5% 提升到 90%
   - DPO 只能优化已有能力，不能学习新能力

2. **SFT 也失败了**
   - 20 个样本太少（需要 100-500）
   - LoRA rank 16 太小（需要 64-128）
   - 模型可能容量不足（1.7B → 4B）

3. **Bug 5: DPO 数据长度不平衡**
   - Rejected 包含 ~2000 tokens 的二进制垃圾
   - 导致模型学习长度偏好而不是工具选择
   - 修复：只保留第一个 assistant turn

4. **vLLM 启动前必须清理显存**
   - 否则会 OOM 或启动失败
   - 已更新 `rl/scripts/start_vllm.sh`

---

## 环境配置

### 1. RunPod 实例要求

- **GPU**: L40S (46GB) 或 A100 (80GB)
- **系统**: Ubuntu 22.04 + CUDA 12.8
- **Python**: 3.12
- **存储**: 至少 50GB（模型 + checkpoints）

### 2. SSH 配置

```bash
# ~/.ssh/config
Host runpod-new
    HostName <POD_IP>
    Port <POD_PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

### 3. 依赖安装

```bash
# 在 RunPod 上执行
cd ~/pinchbench-skill

# 安装 Python 依赖
pip install torch transformers peft trl datasets accelerate bitsandbytes

# 安装 vLLM
pip install vllm

# 验证安装
python -c "import torch; print(torch.cuda.is_available())"
python -c "import vllm; print('vLLM OK')"
```

---

## 训练脚本

### SFT 训练脚本

**文件**: `rl/train/train_sft_lora.py`

**推荐数据集**: `sft_focused_augmented.jsonl` (54 samples)
- 每个样本都是：user prompt → 直接 exec+pandas → 成功
- 不包含探索步骤（ls, read csv 等）
- 通过变体增强到 54 个样本

**关键配置**:
```python
# 模型
--model-name Qwen/Qwen3-1.7B

# 数据（使用 focused 数据）
--data-path rl/data/generated/task_18_spreadsheet_summary_runtime/sft_focused_augmented.jsonl

# LoRA 配置
--lora-rank 64          # 增加到 64（之前 16 太小）
--lora-alpha 128        # alpha = 2 * rank
--max-length 2048       # 避免 OOM

# 训练配置
--num-epochs 5          # 增加到 5 epochs
--batch-size 1
--grad-accum 8          # 有效 batch size = 8
--lr 5e-5
--bf16
```

**完整命令**:
```bash
python3 rl/train/train_sft_lora.py \
  --model-name Qwen/Qwen3-1.7B \
  --data-path rl/data/generated/task_18_spreadsheet_summary_runtime/sft_focused_augmented.jsonl \
  --output-dir rl/checkpoints/sft_focused_v2 \
  --num-epochs 5 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-5 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --max-length 2048 \
  --bf16
```

**数据生成脚本**:
```bash
# 从 teacher rollouts 提取 focused samples
python3 rl/data/scripts/create_focused_sft_data.py

# 通过变体增强数据
python3 rl/data/scripts/augment_focused_data.py
```

**注意事项**:
- ✅ 已启用 gradient checkpointing
- ✅ 已手动 mask 非 assistant tokens
- ✅ 使用 focused 数据（直接 xlsx → pandas）
- ⚠️ rank 64 可能仍会 OOM，如果 OOM 降到 32
- ⚠️ max_length 2048 会截断长轨迹

### DPO 训练脚本

**文件**: `rl/train/train_dpo_lora_fixed.py`

**关键修复**:
```python
# Bug 5 修复：只保留第一个 assistant turn
chosen_completion = [chosen_msgs[1]]    # 只要 assistant message
rejected_completion = [rejected_msgs[1]]  # 不要 tool result
```

**完整命令**:
```bash
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
```

---

## Merge LoRA

**在 CPU 上执行**（不需要 GPU）:

```bash
python3 -c '
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", device_map="cpu", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, "rl/checkpoints/sft_teacher_v1")

print("Merging...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("rl/checkpoints/sft_teacher_v1_merged")
tokenizer.save_pretrained("rl/checkpoints/sft_teacher_v1_merged")

print("✅ Merge 完成！")
'
```

---

## vLLM 启动

### 使用标准脚本（推荐）

**文件**: `rl/scripts/start_vllm.sh`

**已包含**:
- ✅ 自动清理 GPU 显存
- ✅ 验证显存是否足够
- ✅ 正确的参数配置

**用法**:
```bash
# 启动 merged 模型
bash rl/scripts/start_vllm.sh rl/checkpoints/sft_teacher_v1_merged

# 或手动指定参数
VLLM_PORT=8000 bash rl/scripts/start_vllm.sh rl/checkpoints/sft_teacher_v1_merged
```

### 手动启动（调试用）

```bash
# Step 1: 清理显存（CRITICAL！）
pkill -9 -f vllm
sleep 2

GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
for pid in $GPU_PIDS; do
    kill -9 $pid
done
sleep 3

# Step 2: 验证显存
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

# Step 3: 启动 vLLM
python3 -m vllm.entrypoints.openai.api_server \
  --model rl/checkpoints/sft_teacher_v1_merged \
  --served-model-name openai-qwen3-1-7b \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code
```

**关键参数**:
- `--enable-auto-tool-choice`: 必须！否则 OpenClaw 会报错
- `--tool-call-parser hermes`: 必须！解析 tool_calls
- `--gpu-memory-utilization 0.95`: L40S 46GB 可以用 0.95

---

## Benchmark 测试

### 在 Mac 本地执行

```bash
# Step 1: 建立 SSH tunnel
ssh -f -N -L 8000:localhost:8000 runpod-new

# Step 2: 验证连接
curl -s http://localhost:8000/v1/models | jq .

# Step 3: 运行 benchmark
python3 scripts/benchmark.py \
  --model openai-qwen3-1-7b \
  --base-url http://localhost:8000/v1 \
  --suite task_18_spreadsheet_summary \
  --runs 1 \
  --output-dir .pinchbench_runs/sft_teacher_v1
```

### 检查结果

```bash
# 查看分数
grep "Final score" .pinchbench_runs/sft_teacher_v1/*.log

# 查看工具使用
grep -o '"name":"[^"]*"' .pinchbench_runs/sft_teacher_v1/*_transcripts/*.jsonl | sort | uniq -c
```

---

## 常见问题

### 1. OOM (Out of Memory)

**症状**: `torch.OutOfMemoryError: CUDA out of memory`

**原因**:
- LoRA rank 太大
- max_length 太大
- batch_size 太大
- 显存没清理干净

**解决方案**:
```bash
# 降低 rank
--lora-rank 16  # 从 32 降到 16

# 降低 max_length
--max-length 2048  # 从 4096 降到 2048

# 增加 gradient accumulation
--grad-accum 16  # 从 4 增加到 16

# 清理显存
pkill -9 python3
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9
```

### 2. vLLM 启动失败

**症状**: `ValueError: Free memory on device cuda:0 (0.75/44.4 GiB)`

**原因**: 显存没清理干净

**解决方案**:
```bash
# 使用标准脚本（已包含清理）
bash rl/scripts/start_vllm.sh <model_path>

# 或手动清理
pkill -9 -f vllm
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9
sleep 3
```

### 3. Benchmark 报错 "auto tool choice requires..."

**症状**: `400 "auto" tool choice requires --enable-auto-tool-choice`

**原因**: vLLM 启动时缺少参数

**解决方案**:
```bash
# 确保包含这两个参数
--enable-auto-tool-choice \
--tool-call-parser hermes
```

### 4. 训练 loss 不下降

**可能原因**:
- 数据量太少（< 50 samples）
- Learning rate 太小或太大
- LoRA rank 太小
- 数据质量有问题

**调试方法**:
```bash
# 检查数据量
wc -l rl/train/runtime_teacher_rollouts_train.jsonl

# 检查第一个样本
head -1 rl/train/runtime_teacher_rollouts_train.jsonl | jq .

# 检查训练日志
tail -f sft_v1_training.log | grep loss
```

---

## 下一步计划

### 方案 A: 增加数据量 + 更大的 rank

**目标**: 用足够的数据和容量激活 exec 能力

**步骤**:
1. 生成 100+ SFT 样本（从 teacher rollouts）
2. 使用 rank 64（需要更大的 GPU 或优化）
3. 训练 5 epochs
4. Benchmark 验证

**预期**: 可能有效，但不确定

### 方案 B: 换更大的模型

**目标**: 用 Qwen3-4B 代替 1.7B

**步骤**:
1. 测试 Qwen3-4B baseline
2. 如果 baseline 就会用 exec，直接用
3. 如果不会，用 SFT 训练（rank 64）

**预期**: 更有可能成功

### 方案 C: 分析数据质量

**目标**: 确认 teacher 数据是否正确

**步骤**:
1. 检查 20 个 teacher samples
2. 验证是否都展示了 exec+pandas
3. 检查是否有错误的样本

**预期**: 可能发现数据问题

---

## 文件清单

### 训练脚本
- `rl/train/train_sft_lora.py` - SFT 训练（已修复 OOM）
- `rl/train/train_dpo_lora_fixed.py` - DPO 训练（Bug 5 修复）

### 启动脚本
- `rl/scripts/start_vllm.sh` - vLLM 启动（已添加显存清理）

### 数据文件
- `rl/data/generated/task_18_spreadsheet_summary_runtime/runtime_teacher_rollouts_train.jsonl` - SFT 数据（20 samples）
- `rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_pure_focused_train.jsonl` - DPO 数据（16 pairs）

### 分析文档
- `rl/data/generated/task_18_spreadsheet_summary_runtime/分析.md` - 完整的实验分析
- `HANDOFF.md` - 项目交接文档（需要更新）

---

## 关键经验教训

1. **DPO 不能从零激活新能力**
   - 需要 baseline 至少 50% 的能力
   - 只能优化已有能力的偏好

2. **SFT 需要足够的数据和容量**
   - 数据量: 100-500 samples
   - LoRA rank: 64-128
   - 模型大小: 4B+

3. **vLLM 启动前必须清理显存**
   - 否则会 OOM
   - 已集成到启动脚本

4. **训练指标完美 ≠ 模型学会了**
   - Loss → 0 不代表学到了正确的东西
   - 必须通过 benchmark 验证实际能力

5. **LoRA rank 16 太小**
   - 无法学习复杂的新能力
   - 建议至少 64

6. **数据质量 > 数据量**
   - 但数量也很重要
   - 20 个样本远远不够
