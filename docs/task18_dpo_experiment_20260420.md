# task_18 DPO 实验记录（2026-04-20）

## 背景

在 `docs/spreadsheet_data_production_module.md` 完成模块设计后，本轮目标是把整个 DPO 流程跑通并得到可对比的 benchmark 数字：

- 用 **qwen3.6-plus** 作为 teacher（替代之前的 qwen-plus，原因见下）
- 用 **Qwen3-1.7B** 作为 student（本地 vLLM，Mac）
- 构造干净的 DPO pairs（解决绝对路径污染问题）
- 训练 LoRA DPO（RunPod A100）
- benchmark 对比 base vs DPO

---

## 1. Teacher 模型选择：qwen3.6-plus

### 为什么换 teacher

之前用 qwen-plus 采集的 teacher rollout 问题：
- qwen-plus mean_score=0.915，不是满分
- 部分样本 turns 高达 23 轮，轨迹冗长
- 这些"chosen"轨迹质量不够，教错模型

qwen3.6-plus 的表现：
- 20/20 variants 全部满分（mean_score=1.0）
- 平均 4.7 turns 完成任务
- 轨迹简洁，工具调用有效

### lib_agent.py 路由修复

qwen3.6-plus 在 OpenClaw 里路由到了错误的 provider，导致 401：

```python
# 修复前：走 qwen provider → auth 401
# 修复后（scripts/lib_agent.py）：
if mid in {"qwen-plus", "qwen3.6-plus"}:
    return "dashscope", mid
```

同时在 `~/.openclaw/openclaw.json` 的 dashscope provider models 列表里加上 `qwen3.6-plus`。

---

## 2. 数据采集：绝对路径 Bug 与修复

### Bug 发现

第一轮 DPO 训练（v1）benchmark 提升有限，检查轨迹发现 teacher 的 exec 命令里有硬编码绝对路径：

```bash
# 训练数据里的命令（错误）
cd /tmp/pinchbench_spreadsheet_dpo_qwen36/9100/agent_workspace && python3 -c "..."

# 模型推理时的 workspace（不同路径）
/tmp/pinchbench/0141/agent_workspace/
```

模型在推理时看到的 workspace 路径和训练数据完全不同，学到了无法泛化的路径模式。

### 修复：`_strip_workspace_path`

在 `rl/data_construction/collect_spreadsheet_runtime_rollouts.py` 里加入路径剥离：

```python
def _strip_workspace_path(text: str, workspace: str) -> str:
    """Replace absolute workspace/run paths with '.' so paths generalise at inference."""
    # Strip agent_workspace and its parent (run_dir):
    #   /tmp/.../9100/agent_workspace/foo.csv  → foo.csv
    #   /tmp/.../9100/                          → ./
    for path in [workspace, str(Path(workspace).parent)]:
        p = path.rstrip("/")
        text = text.replace(p + "/", "")
        text = text.replace(p, ".")
    return text
```

同时加了 regex 兜底（处理 `read` tool 的 `path` 字段，这类工具参数不走 `exec` 路径）：

```python
AGENT_WS_RE = re.compile(r'/tmp/[^/]+/\d+/agent_workspace/')
# post-process 时对已有数据统一清洗
```

**应用位置**：
- `tool_calls[*].function.arguments`（exec/read 的 JSON 参数）
- `tool_result.content`（工具返回内容里的路径引用）

### 采集配置

**Teacher（qwen3.6-plus via Dashscope）：**
```bash
DASHSCOPE_API_KEY=... python3 rl/data_construction/collect_spreadsheet_runtime_rollouts.py \
  --input rl/data/generated/task_18_spreadsheet_summary_runtime/rl_train.jsonl \
  --output ...runtime_teacher36plus_clean_rollouts_train.jsonl \
  --role teacher \
  --model qwen3.6-plus \
  --timeout-seconds 300 \
  --run-root /tmp/pinchbench_dpo_clean
```

**Student（Qwen3-1.7B via 本地 vLLM port 18021）：**
```bash
python3 rl/data_construction/collect_spreadsheet_runtime_rollouts.py \
  --input rl/data/generated/task_18_spreadsheet_summary_runtime/rl_train.jsonl \
  --output ...runtime_student_clean_rollouts_train.jsonl \
  --role student \
  --model Qwen3-1.7B \
  --base-url http://127.0.0.1:18021/v1 \
  --timeout-seconds 240 \
  --run-root /tmp/pinchbench_dpo_clean_student
```

### 采集结果

| 角色 | 模型 | 样本数 | mean_score | avg_turns |
|------|------|--------|-----------|-----------|
| Teacher | qwen3.6-plus | 20/20 | 0.995 | 4.7 |
| Student | Qwen3-1.7B | 20/20 | 0.155 | 8.7 |

Student 平均 8.7 turns（vs teacher 4.7 turns）是正常的——student 不会做任务，会反复尝试失败。

---

## 3. DPO Pairs 构建

```bash
python3 rl/data_construction/build_dpo_pairs.py \
  --prompts rl/data/generated/task_18_spreadsheet_summary_runtime/rl_train.jsonl \
  --teacher-rollouts ...runtime_teacher36plus_clean_rollouts_train.jsonl \
  --student-rollouts ...runtime_student_clean_rollouts_train.jsonl \
  --output ...dpo_pairs_clean_train.jsonl
```

筛选规则：
- chosen：teacher score >= 0.9，assistant_turns <= 8
- rejected：student score <= 0.5

结果：**20 pairs，0 skipped**（chosen 全满分，rejected 全失败，无边界案例）

输出文件：`rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_clean_train.jsonl`

---

## 4. DPO 训练（RunPod A100）

### 配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen/Qwen3-1.7B |
| 方法 | DPO + LoRA（TRL 1.2.0） |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| target modules | q/k/v/o_proj, gate/up/down_proj |
| lr | 1e-5 |
| epochs | 3 |
| batch_size | 1 |
| grad_accum | 4 |
| max_length | 4096 |
| beta | 0.1 |
| dtype | bf16 |

### 训练脚本

```bash
# rl/train/run_dpo_lora.sh
python3 rl/train/train_dpo_lora.py \
  --model-name Qwen/Qwen3-1.7B \
  --data-path .../dpo_pairs_clean_train.jsonl \
  --output-dir .../dpo_lora_qwen31_task18_v2 \
  --num-epochs 3 --batch-size 1 --grad-accum 4 \
  --lr 1e-5 --lora-rank 32 --lora-alpha 64 \
  --max-length 4096 --beta 0.1 --bf16
```

### 训练过程

训练耗时约 **2 分钟**（20 samples × 3 epochs = 15 steps）。

关键指标趋势：

| step | epoch | loss | rewards/margins | rewards/accuracies |
|------|-------|------|-----------------|-------------------|
| 1 | 0.21 | 0.693 | 0 | 0 |
| 5 | 1.0 | 2.8e-5 | 10.5 | 1.0 |
| 10 | 2.0 | 6.0e-8 | 24.8 | 1.0 |
| 15 | 3.0 | 4.1e-8 | 28.3 | 1.0 |

**train_loss=0.054**，rewards/margins 最终约 28，chosen/rejected 分离明显。

> 注意：loss 在 step 4-5 就接近 0，说明 20 样本对 LoRA rank-32 来说可能过拟合。

### Checkpoint

`/workspace/pinchbench-skill/rl/checkpoints/dpo_lora_qwen31_task18_v2/`

```text
adapter_config.json
adapter_model.safetensors   # 139MB
tokenizer.json
tokenizer_config.json
train.log
checkpoint-10/
checkpoint-15/
```

### 踩坑：rsync 路径写错

第一次 rsync 把 `run_dpo_lora.sh` 传到了 data 目录而不是 `rl/train/`，导致 pod 上的脚本没更新，第一次训练跑的是旧数据（v1 paths）存到旧目录（task18 非 v2）。

修复命令：
```bash
rsync -avz -e "ssh -p 28610 ..." \
  rl/train/run_dpo_lora.sh rl/train/train_dpo_lora.py \
  root@195.26.232.162:/workspace/pinchbench-skill/rl/train/
```

---

## 5. vLLM 启动配置

关键参数（踩坑后的完整配置）：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --served-model-name Qwen3-1.7B \      # 必须，否则 OpenClaw 找不到模型
  --enable-lora \
  --lora-modules dpo-v2=/path/to/dpo_lora_qwen31_task18_v2 \
  --max-lora-rank 32 \
  --port 18021 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \           # 必须，否则 tool choice=auto 报 400
  --tool-call-parser hermes \           # 必须，否则同上
  --override-generation-config '{"enable_thinking": false}'  # 必须，否则 think 标签破坏 content 解析
```

第一次忘了加 `--enable-auto-tool-choice` 和 `--tool-call-parser`，导致所有请求 400 错误，score=0.0。

---

## 6. Benchmark 结果

### task_18_spreadsheet_summary，3 runs each

| 模型 | run 1 | run 2 | run 3 | mean |
|------|-------|-------|-------|------|
| Base Qwen3-1.7B | 0.192 | 0.025 | 0.025 | **0.081** |
| DPO v2 (clean) | 0.258 | 0.142 | 0.125 | **0.175** |

DPO v2 vs base：**+115%**（0.081 → 0.175）

### 分项得分（DPO v2 run 1）

| 维度 | 得分 |
|------|------|
| report_created | 1.0 |
| budget_comparison | 1.0 |
| total_revenue | 0.0 |
| total_profit | 0.0 |
| top_region | 0.0 |
| top_product | 0.0 |
| total_expenses | 0.0 |
| top_department | 0.0 |
| top_employee | 0.0 |

### 分析

DPO 学会了"该做什么"（创建文件、写 budget/actual 结构），但没学会"怎么算对"：

- 所有数字维度得分仍为 0
- 核心瓶颈：xlsx 解析失败——模型要么读出 binary garbage，要么用错方法，算出的数字完全不对
- 20 samples 的 DPO 能教会行为模式，但精确数值计算需要更多数据或 RL 过程奖励

---

## 7. 历史对照

| 版本 | 数据 | mean score | 备注 |
|------|------|-----------|------|
| Base | - | 0.081 | 3-run avg |
| DPO v1 | qwen-plus teacher，绝对路径 | ~0.18 | 3次单跑均值 |
| DPO v2 | qwen3.6-plus teacher，干净路径 | 0.175 | 3-run avg |

v1 和 v2 接近，说明路径修复后效果相当，核心瓶颈不在路径泛化，而在数值计算能力。

---

## 8. 下一步

### 方向 A：更多 DPO 数据（扩量）

当前 20 samples 太少，loss 在 5 steps 后就崩到 0，明显过拟合。

建议扩到 80-100 pairs：
- chosen 筛选放宽：turns <= 10（qwen3.6-plus 的 turns 已经很短）
- 补充不同 prompt variant 的样本（当前 20 个 variant 覆盖了所有 5 种 prompt）

### 方向 B：RL 训练

DPO warm-up 之后接 REINFORCE：
- 使用 `rl_train.jsonl` 的 20 个 synthetic variants 作为环境
- reward = 数字准确性（score on each breakdown dimension）
- process reward = no repeated xlsx binary read, 8 turns 内完成
- 参考现有 `rl/train/run_reinforce_lora.sh` 配置

### 方向 C：分析 exec 轨迹

在做更多训练之前，先 trace 一条 student 轨迹，确认：
1. model 发的 exec 命令是否正确（read CSV/xlsx）
2. exec 是否成功运行（exit code）
3. model 是否正确解读了 exec 的输出（还是忽略了数字）

---

## 附：关键文件路径

| 文件 | 说明 |
|------|------|
| `rl/data_construction/collect_spreadsheet_runtime_rollouts.py` | 采集 teacher/student rollouts，含路径剥离 |
| `rl/data_construction/build_dpo_pairs.py` | 配 chosen/rejected pairs |
| `rl/train/train_dpo_lora.py` | TRL 1.2.0 DPO + PEFT LoRA 训练脚本 |
| `rl/train/run_dpo_lora.sh` | RunPod 训练启动脚本 |
| `scripts/lib_agent.py` | OpenClaw provider/model 路由（含 qwen3.6-plus 修复） |
| `rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_clean_train.jsonl` | 最终干净 DPO 数据（20 pairs） |
| `/workspace/pinchbench-skill/rl/checkpoints/dpo_lora_qwen31_task18_v2/` | RunPod DPO v2 checkpoint |
