# PinchBench RL

用 PinchBench 的 task 作为训练环境，构建 GRPO 训练数据。

## 核心思路

PinchBench 的每个 task 天然满足 RL 训练的要求：

- **真实 tool call 环境**：通过 openclaw agent 执行，不是 mock
- **确定性 reward**：automated grading 函数直接出 0/1 信号
- **workspace 隔离**：每次执行独立，没有副作用，支持多次采样
- **Live-User 场景**：task prompt 就是真实用户请求

```
task prompt（原始或变体）
    → openclaw agent 执行 K 次（每次独立 workspace）
    → grading 函数打 terminal reward [0, 1]
    → K 条 trajectory + reward → GRPO 组内对比训练
```

## 目录结构

```
rl/
  README.md               # 本文件
  schema.py               # 训练数据格式定义
  collect.py              # 采样脚本：对指定 task 跑 K 次，收集 trajectory
  convert.py              # 转换：openclaw transcript → 训练格式
  analyze.py              # 分析 benchmark 结果，输出 RL 难度分布
  tasks/                  # 训练专用 task 变体（不污染测试集）
    task_21_comprehension_train_*.md
    ...
```

## 数据格式（TrainingSample）

每条训练样本是一个 JSON 对象，一个 task 的 K 次采样打包在一起：

```json
{
  "sample_id": "task_21_openclaw_comprehension-seed42-run0",
  "task_id": "task_21_openclaw_comprehension",
  "split": "train",
  "seed": 42,
  "model_id": "qwen-plus",
  "prompt": "What is the capital of France?...",
  "grading_type": "automated",
  "trajectory": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "...",
      "tool_calls": [
        {"name": "bash", "arguments": {"command": "echo hello"}}
      ],
      "logprobs": null
    },
    {
      "role": "tool",
      "tool_name": "bash",
      "content": "hello"
    },
    {
      "role": "assistant",
      "content": "Done."
    }
  ],
  "reward": {
    "terminal": 0.8,
    "breakdown": {
      "criterion_a": 1.0,
      "criterion_b": 0.6
    }
  },
  "usage": {
    "input_tokens": 12000,
    "output_tokens": 300,
    "total_tokens": 12300
  },
  "execution_time": 25.3,
  "timed_out": false
}
```

### 字段说明

| 字段 | 含义 |
|------|------|
| `sample_id` | 唯一 ID，`{task_id}-seed{seed}-run{i}` |
| `task_id` | 对应 PinchBench task ID |
| `split` | `train` / `val` / `test`（test 永远不进训练） |
| `seed` | task 变体的 seed，控制 train/val/test 划分 |
| `model_id` | 采样时使用的模型 |
| `prompt` | 发给 agent 的完整 prompt |
| `grading_type` | `automated` / `llm_judge` / `hybrid` |
| `trajectory` | 完整对话轮次，包含 tool calls |
| `trajectory[].logprobs` | 每个 token 的 log prob，GRPO 训练必需；采样时记录 |
| `reward.terminal` | grading 函数输出的 [0,1] 分数，作为 terminal reward |
| `reward.breakdown` | 各评分维度的细项分数 |

### Split 划分

seed 范围决定 split（与 clawgym 保持一致）：

| Split | Seed 范围 | 用途 |
|-------|-----------|------|
| train | [0, 10000) | 训练 |
| val | [10000, 11000) | 验证，监控过拟合 |
| test | [11000, 12000) | 禁止训练，仅评估 |

PinchBench 原始 25 个 task 对应 test split，**永远不进训练**。

训练数据来自 `rl/tasks/` 下的变体 task（相同类型，不同 prompt 措辞）。

## 快速开始

### 1. 分析 benchmark 结果，找 RL 目标 task

```bash
python rl/analyze.py results/0004_qwen-plus.json
```

### 2. 对目标 task 采样 K 次

```bash
python rl/collect.py \
  --task task_21_openclaw_comprehension \
  --runs 8 \
  --model qwen-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-key YOUR_KEY \
  --output rl/data/
```

### 3. 查看采集结果

```bash
python rl/analyze.py rl/data/task_21_openclaw_comprehension.jsonl
```

## RL 算法

使用 **GRPO**（Group Relative Policy Optimization）：

- 同一 prompt 采样 K 条 trajectory
- terminal reward 来自 grading 函数
- 组内归一化 advantage：`A_k = (r_k - mean(r)) / std(r)`
- 不需要 critic，不需要 reference model 显式 KL（由 clip 控制）

适合 PinchBench 场景的原因：
- workspace 隔离，K 次采样互不干扰
- grading 是确定性函数，reward 无噪声（automated 类型）
- terminal reward 粒度够用，不需要 per-step dense reward

## 注意事项

1. **logprobs 必须在采样时记录**：训练框架（veRL/trl）需要 old log prob 计算 importance ratio。推理用 vLLM，设置 `logprobs=True`。

2. **只用 automated grading 的 task 做 RL 主力**：llm_judge reward 有噪声，适合辅助分析，不适合直接做训练信号。

3. **变体 task 的 grading 函数要跟着改**：换了人名/时间/参数，grading 里的 regex 也要对应更新。
