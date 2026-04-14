# PinchBench RL

用 PinchBench 的 task 模拟 Live-User 场景，构建 single-sample PG 训练数据。

## 核心思路

PinchBench 的每个 task 对应一个真实用户请求场景：

- **task prompt = 用户请求**：日历、邮件、研究、写作等真实任务
- **openclaw 执行 = 真实 agent 交互**：不回滚，不多采样，一次执行一条 trajectory
- **grading 函数 = terminal reward**：automated 类型有确定性 0/1 信号
- **workspace 隔离**：每次执行独立，无副作用

与 **`scripts/benchmark.py` 评测对比**时，请让 OpenClaw 侧的 **`MAX_TURNS`** 与训练一致（见 `rl/train/run_reinforce_lora.sh`，默认 **10**）。Benchmark 以任务 `timeout_seconds` 控制 wall-clock；RL 里 `OpenClawAgentLoop` 还会用 `MAX_TURNS` 限制经 proxy 的模型轮数，两者不一致会导致训/测不可比。

```
task prompt（用户请求）
    → openclaw + Qwen3-4B 执行一次
    → transcript（真实 tool call 序列）
    → grading 打 terminal reward [0, 1]
    → vLLM re-score 补 logprobs
    → TrainingSample → single-sample PG 训练
```

## 技术栈

| 组件 | 说明 |
|------|------|
| openclaw | agent 执行环境（本地 Mac） |
| Qwen3-4B | 训练目标模型（开源权重） |
| vLLM | 推理服务 + re-score logprobs（RunPod） |
| veRL | RL 训练框架（RunPod A100） |
| PinchBench | 采样环境 + 评测集 |

## 目录结构

```
rl/
  README.md               # 本文件
  schema.py               # 训练数据格式（TrainingSample）
  collect.py              # 采样：跑 PinchBench task，收集 transcript
  convert.py              # 转换：openclaw transcript → TrainingSample
  rescore.py              # re-score：vLLM 补全 logprobs
  analyze.py              # 分析 benchmark 结果，输出 RL 难度分布
  train/
    train.py              # veRL single-sample PG 训练主脚本
    reward.py             # reward 计算（terminal + immediate）
    data.py               # 训练数据加载
  scripts/
    start_vllm.sh         # RunPod 上启动 vLLM serve
    setup_new_pod.sh      # RunPod 新 Pod 环境初始化（唯一真实入口）
  tasks/                  # 训练专用 task 变体（不污染测试集）
    README.md
```

## 数据格式（TrainingSample）

```json
{
  "sample_id": "task_21_openclaw_comprehension-seed42-run0",
  "task_id": "task_21_openclaw_comprehension",
  "split": "train",
  "seed": 42,
  "run_index": 0,
  "model_id": "Qwen/Qwen3-4B",
  "prompt": "...",
  "grading_type": "automated",
  "trajectory": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "...",
      "tool_calls": [{"name": "bash", "arguments": {"command": "..."}}],
      "logprobs": [-0.12, -0.45, ...]
    },
    {
      "role": "tool",
      "tool_name": "bash",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "Done.",
      "logprobs": [-0.08, ...]
    }
  ],
  "reward": {
    "terminal": 0.8,
    "breakdown": {"criterion_a": 1.0, "criterion_b": 0.6}
  },
  "usage": {"input_tokens": 12000, "output_tokens": 300, "total_tokens": 12300},
  "execution_time": 25.3,
  "timed_out": false
}
```

### 字段说明

| 字段 | 含义 |
|------|------|
| `sample_id` | 唯一 ID，格式 `{task_id}-seed{seed}-run{i}` |
| `split` | `train` / `val` / `test`（test 不进训练） |
| `seed` | task 变体 seed，控制 train/val/test 划分 |
| `trajectory[].logprobs` | vLLM re-score 后补全，训练必需 |
| `reward.terminal` | grading 分数 [0,1]，广播到所有 assistant turn |

### Split 划分

| Split | Seed 范围 | 用途 |
|-------|-----------|------|
| train | [0, 10000) | 训练 |
| val | [10000, 11000) | 验证 |
| test | [11000, 12000) | 评估（禁止训练） |

**PinchBench 原始 25 个 task = test split，永远不进训练。**
训练数据来自 `rl/tasks/` 下的变体 task。

## RL 算法：Single-Sample PG

参考 `docs/rl-algorithm.md`（Phase 1）。

### Reward 分解

```
r_t = r_immediate + r_terminal（广播）

r_immediate:
  - 格式错误 / 幻觉 / 工具参数非法 → -1（直接惩罚）
  - 格式正常 → 0（看 terminal）

r_terminal:
  - grading 函数输出 [0, 1]
  - 广播到该 episode 所有 assistant turn
```

### 训练 Loss

```
L = L_PG + β_kl * KL(π_θ || π_ref) + c_critic * L_critic

L_PG = -Σ min(ρ_i * A_t, clip(ρ_i, 1-ε, 1+ε) * A_t)
ρ_i = exp(logp_new_i - logp_old_i)
A_t = r_t - V_φ(h_t)   # critic baseline 降方差
```

## 快速开始

### Step 1：RunPod 起 vLLM

```bash
# RunPod 上执行
export ECS_HOST=<阿里云 ECS 公网 IP>
bash rl/scripts/setup_new_pod.sh
bash rl/scripts/start_vllm.sh Qwen/Qwen3-4B
```

### Step 2：配置 openclaw 接 vLLM

```bash
# 本地 Mac，把 openclaw 的模型换成 RunPod vLLM
python rl/collect.py --setup \
  --base-url http://<runpod-ip>:8000/v1 \
  --model Qwen/Qwen3-4B
```

### Step 3：采样 training task

```bash
# 对 rl/tasks/ 下的变体 task 各跑一次
python rl/collect.py \
  --tasks-dir rl/tasks \
  --base-url http://<runpod-ip>:8000/v1 \
  --model Qwen/Qwen3-4B \
  --output rl/data/samples_raw.jsonl
```

### Step 4：vLLM re-score 补 logprobs

```bash
# RunPod 上执行
python rl/rescore.py \
  --input rl/data/samples_raw.jsonl \
  --output rl/data/samples_rescored.jsonl \
  --model Qwen/Qwen3-4B \
  --base-url http://localhost:8000/v1
```

### Step 5：veRL 训练

```bash
# RunPod 上执行
python rl/train/train.py \
  --data rl/data/samples_rescored.jsonl \
  --model Qwen/Qwen3-4B \
  --output rl/checkpoints/
```

### Step 6：PinchBench 评测

```bash
# 本地 Mac，更新 openclaw 用新 checkpoint
./scripts/run.sh \
  --model Qwen3-4B-finetuned \
  --base-url http://<runpod-ip>:8000/v1 \
  --no-upload \
  --suite automated-only
```

## 注意事项

1. **只用 automated grading 的 task 做 RL 主力**：reward 确定性强，无噪声
2. **logprobs 必须 re-score 后才能训练**：openclaw 不暴露 token logprobs
3. **变体 task 的 grading 函数要同步更新**：换了参数，regex 也要改
4. **分析 hard bucket 再决定训练哪些 task**：`python rl/analyze.py results/latest.json`
