# PinchBench Online RL 算法设计文档

> 最后更新: 2026-04-09

---

## 1. 问题定义

### 1.1 目标

用强化学习训练 Qwen3-4B（小模型），使其在 PinchBench 实际编程任务上的表现逐步提升。模拟 **live-user 场景**：每次交互只有一次机会（rollout.n=1），没有多次重试的条件。

### 1.2 为什么不用离线 RL / 预采集数据？

| 方案 | 问题 |
|------|------|
| 离线采集 + GRPO | GRPO 需要同一 prompt 的多次 rollout 计算相对优势，不适用于 live-user 的 n=1 场景 |
| 离线采集 + PG | rollout.n=1 没有 baseline，梯度方差极大，学不动 |
| **Online RL（当前方案）** | 每个 training step 用当前策略产生新数据，拿到 reward 立即更新 |

### 1.3 核心挑战

Qwen3-4B 在 8 个选定 task 上的 baseline 成功率大部分是 **0%**。如果只用 terminal reward（成功/失败），模型永远拿到 -1，学不到任何有用信号。

**解决方案**：Process Reward Model (PRM) — 对每个 turn 独立打分，告诉模型"这一步做对了"。

---

## 2. 系统架构

```
┌─── RunPod (L4 GPU, 24GB) ─────────────┐
│                                         │
│  veRL Trainer (REINFORCE + LoRA)        │
│    ├── Actor: Qwen3-4B + LoRA (FSDP)   │
│    ├── Ref Model: Qwen3-4B (CPU)       │
│    ├── vLLM Rollout Engine (半卡 GPU)   │
│    │     ├── Agent 推理（做任务）        │
│    │     └── PRM 推理（打分）           │
│    └── OpenClawAgentLoop                │
│          └── ModelProxy (:PORT)         │
│                                         │
└─────────────┬───────────────────────────┘
              │ SSH + HTTP
              ▼
┌─── 阿里云 ECS (4核 8G) ───────────────┐
│                                         │
│  OpenClaw Agent (--local)               │
│    ├── 工具: bash, read, write, edit    │
│    ├── web_search, web_fetch            │
│    └── models.json → ModelProxy         │
│                                         │
│  PinchBench Grading                     │
│    ├── automated: Python 脚本检查       │
│    └── llm_judge: qwen-plus API         │
│                                         │
└─────────────────────────────────────────┘
```

---

## 3. 算法细节

### 3.1 训练算法：REINFORCE + LoRA

- **REINFORCE**：rollout.n=1 的 live-user 场景，每次交互只有一次机会，无法做 group rollout
- **LoRA**（rank=32, alpha=64, target=all-linear）：参数高效微调，单卡 L4 可跑
- 不需要 Critic 网络（process reward 已提供密集信号）
- 不需要 baseline：process reward 自带正负方向，terminal reward 用 {-1, +1}

### 3.2 一个 Training Step 的完整流程

```
Step 1: Rollout（攒一个 batch）
  ├── 从 8 个 task 中采样 batch_size=8 个（每个 task 1 条）
  ├── 并发 4 个 OpenClaw episode（ECS 4核限制）
  ├── 每个 episode:
  │     ├── OpenClaw agent 用当前 LoRA 策略执行 task
  │     ├── ModelProxy 拦截 LLM 请求 → vLLM 生成
  │     ├── OpenClaw 执行工具 → 下一轮对话
  │     ├── episode 结束 → PinchBench grading → terminal reward
  │     └── 对每个 assistant turn → Qwen3-4B self-judge → process reward
  └── 收集 8 条 (prompt_ids, response_ids, mask, rewards)

Step 2: Training（一次参数更新）
  ├── 计算 advantages（直接用 reward，不跨 task 做均值/std）
  ├── REINFORCE policy gradient loss on batch
  ├── LoRA 参数更新（ppo_epochs=2）
  └── KL penalty（coef=0.05, 防止偏离 ref 太远）

Step 3: Sync
  └── 新 LoRA 权重 → vLLM rollout engine（layered_summon）
```

### 3.3 Reward 设计

#### Terminal Reward

```
terminal_reward = +1  (PinchBench grading 通过, score >= 0.5)
terminal_reward = -1  (PinchBench grading 未通过)
```

- **不用 {0, 1}，用 {-1, +1}**：天然有方向信号，失败时负梯度推动策略远离当前行为
- 不需要 baseline 或 EMA，±1 自带方向

#### Process Reward (PRM)

```
process_reward[turn_k] ∈ [-0.5, +0.3]
```

- **独立于 terminal reward**：即使 terminal=-1，process reward 仍然生效
- 这是关键设计——Qwen3-4B 成功率 0% 的 task，只靠 terminal 学不动，必须靠 process reward 指导"哪步做对了"

#### 最终 reward 分配

```
reward[turn_1] = process_reward[1]         # e.g., +0.15
reward[turn_2] = process_reward[2]         # e.g., +0.10
...
reward[turn_k] = process_reward[k]         # e.g., +0.25
reward[last]   = process_reward[last] + terminal_reward  # e.g., +0.00 + 1.0
```

token 级别：reward 放在每个 assistant turn 的 `<|im_end|>` token 位置。

#### 举例

成功的 episode (task_02_stock):

| Turn | 行为 | Process | Terminal | 总 Reward |
|------|------|:-------:|:--------:|:---------:|
| 1 | web_search 搜索股价 | +0.15 | | +0.15 |
| 2 | web_fetch 获取详情 | +0.10 | | +0.10 |
| 3 | 分析数据 | +0.00 | | +0.00 |
| 4 | write stock_report.txt | +0.25 | | +0.25 |
| 5 | 总结确认 | +0.00 | +1.0 | +1.00 |

失败但过程部分正确:

| Turn | 行为 | Process | Terminal | 总 Reward |
|------|------|:-------:|:--------:|:---------:|
| 1 | web_search 正确搜索 | +0.15 | | +0.15 |
| 2 | 不写文件就停了 | -0.30 | -1.0 | -1.30 |

→ Turn 1 的 +0.15 不会被 terminal 淹没，模型学到"搜索是对的，问题出在后面"

### 3.4 Advantage 计算

**不跨 task 做 baseline**：不同 task 难度不同，reward 分布不同，不能混算均值/std。

```
advantage[turn_k] = reward[turn_k]  # 直接用 reward 作为 advantage
```

理由：
- process reward 自带正负方向（好行为 +, 坏行为 -）
- terminal reward 自带 ±1 方向
- 不需要额外 baseline 来区分好坏

---

## 4. Process Reward Model (PRM) 设计

### 4.1 Self-Judge 机制

PRM 由 **Qwen3-4B 自己** 担任（self-judge），而非外部强模型。

核心逻辑：
- 做任务时：Qwen3-4B 可能不知道怎么做对（能力不足）
- 做 judge 时：给它详细 rubric + 标准答案，它能判断"这步做得对不对"
- **类比**：学生考试不会做，但给标准答案让他改卷，他是能改的

### 4.2 PRM Prompt 结构

```
你是一个 AI Agent 行为评估器。

## 任务目标
{task_goal — 从 rubric 获取}

## 参考路径（成功 agent 的做法）[天眼]
1. web_search: 搜索 AAPL 股价
2. web_fetch: 获取详细数据（可能多次重试）
3. write: 创建 stock_report.txt，包含价格、日期、摘要
4. 总结确认

## 常见错误
- 搜索后不写文件就停了（过早终止）
- 不搜索就写报告（编造数据）
- 重复失败命令不换策略

## Agent 之前的行为
  Turn 1: web_search(AAPL stock price) → OK
  Turn 2: web_fetch(yahoo finance) → OK

## 当前 Turn (3 of ~4)
工具: write
参数: {"path": "stock_report.txt", "content": "..."}
结果: OK

## 评分
-0.5 到 +0.3 之间打分。
输出 JSON: {"score": <float>, "reason": "..."}
```

### 4.3 天眼（Open-Eye Judge）

rubric 中的"参考路径"来自 **qwen-plus 的成功轨迹**。这是"开天眼"：

- qwen-plus 在这 8 个 task 上成功率高
- 我们提取了它的成功策略：先做什么、后做什么、遇到问题怎么处理
- 写成 rubric，让 Qwen3-4B 拿着标准答案去评判自己的行为

### 4.4 Self-Judge 的自进化

```
训练初期: Qwen3-4B agent 弱, self-judge 也不太准
    ↓ RL 训练
训练中期: agent 稍微变强, self-judge 判断也更准（同一个模型）
    ↓ 正反馈循环
训练后期: agent 变强, self-judge 更准, 学习加速
```

这比固定用 qwen-plus 做 judge 更有意义——如果 qwen-plus 能做得好，为什么不直接用它当 agent？self-judge 的价值在于**自进化**。

### 4.5 PRM 调用开销

- PRM 走 RunPod 上同一个 vLLM 实例（Qwen3-4B）
- 不花 API 费用
- 每个 turn 一次推理，prompt ~500 tokens，output ~50 tokens
- 8 个 task × 平均 8 turns = ~64 次 PRM 调用 / training step
- vLLM batch inference，几秒内完成

---

## 5. 训练配置

### 5.1 模型

| 组件 | 模型 | 位置 |
|------|------|------|
| Agent（策略模型） | Qwen3-4B + LoRA | RunPod vLLM |
| PRM（self-judge） | Qwen3-4B（同上） | RunPod vLLM |
| Ref Model | Qwen3-4B（frozen） | RunPod CPU offload |
| Terminal Judge | qwen-plus | DashScope API |

### 5.2 超参数

| 参数 | 值 | 理由 |
|------|-----|------|
| LoRA rank | 32 | 平衡参数量和表达力 |
| LoRA alpha | 64 | alpha/rank = 2, 标准设置 |
| Learning rate | 2e-5 | LoRA 训练常用值 |
| Batch size | 8 | 8 个 task 各 1 条 |
| PPO epochs | 2 | 每个 batch 上多跑两轮 |
| KL coef | 0.05 | 防止偏离 ref 太远 |
| Entropy coef | 0.01 | 鼓励探索 |
| GPU memory (vLLM) | 45% | 剩余给 Actor training |
| Max turns | 30 | 单个 episode 最大交互轮数 |
| Temperature | 0.7 | rollout 探索温度 |
| Total epochs | 50 | 50 × 8 = 400 个 episode |

### 5.3 硬件

| 资源 | 配置 | 用途 |
|------|------|------|
| RunPod L4 | 24GB VRAM, 50GB RAM | 训练 + vLLM 推理 |
| 阿里云 ECS | 4核 8G, 公网 IP | OpenClaw 工具执行 |
| DashScope API | qwen-plus | Terminal grading judge |

---

## 6. Ablation 实验设计

四组对比实验：

| 实验组 | PRM 方式 | 天眼 | Terminal | 环境变量 |
|-------|---------|:---:|:-------:|---------|
| **A: Baseline** | 无 | ✗ | ✓ | `REWARD_MODE=baseline` |
| **B: Rule-only** | Python 规则 | ✗ | ✓ | `REWARD_MODE=rule` |
| **C: Self-Judge** | Qwen3-4B | ✓ | ✓ | `REWARD_MODE=self-judge` |
| **D: Oracle-Judge** | qwen-plus | ✓ | ✓ | `REWARD_MODE=oracle-judge` |

### 预期结果与论文贡献

| 对比 | 验证什么 | 预期 |
|------|---------|------|
| A vs B | Process reward 是否有用 | B > A（尤其在成功率 0% 的 task 上） |
| B vs C | LLM judge 是否比规则好 | C > B（语义判断更灵活） |
| C vs D | Self-judge 是否足够好 | C ≈ D 或 C 略低（但 C 更有意义） |
| C 的收敛曲线 | Self-judge 是否自进化 | 训练后期收敛加速 |

核心论文贡献：
1. Process Reward 让 terminal=0 的 task 也能学习（A vs B/C）
2. 天眼 reference trajectory 提升 credit assignment 质量（有 vs 无天眼）
3. Self-judge 自进化可行性验证（C 的收敛曲线）

---

## 7. 选定的 8 个训练 Task

| Task ID | 名称 | 类型 | Qwen3-4B Baseline | qwen-plus |
|---------|------|------|:-----------------:|:---------:|
| task_02_stock | Stock Price Research | research | 0% | 100% |
| task_10_workflow | Multi-step API Workflow | complex | 25% | 75% |
| task_12_skill_search | Search & Replace in Files | file_ops | 0% | 100% |
| task_16_email_triage | Email Triage | email | 0% | 50% |
| task_18_market_research | Market Research | research | 0% | 100% |
| task_19_spreadsheet | Spreadsheet Analysis | data | 0% | 50% |
| task_22_second_brain | Knowledge Persistence | knowledge | 0% | 75% |
| task_24_polymarket | Polymarket Briefing | research | 0% | 100% |

选择标准：
- qwen-plus 能做到（证明任务本身可解）
- Qwen3-4B baseline 低或 0%（有提升空间）
- 覆盖不同任务类型（research, file_ops, complex, data）
- 需要多步工具调用（适合 process reward）

---

## 8. 代码结构

```
rl/
├── agent_loop/
│   ├── openclaw_agent_loop.py   # 核心: 驱动 OpenClaw + ModelProxy
│   ├── model_proxy.py           # HTTP 反向代理，拦截 LLM 请求
│   ├── trajectory.py            # token-level 轨迹重建
│   ├── reward.py                # PRM: self-judge + rubric
│   └── config.yaml              # veRL agent loop 注册
├── train/
│   ├── run_reinforce_lora.sh     # 训练启动脚本
│   ├── prepare_prompts.py       # task prompts → veRL parquet
│   └── reward_manager.py        # veRL reward manager 适配
├── scripts/
│   ├── setup_ecs.sh             # 阿里云 ECS 环境初始化
│   └── start_vllm.sh            # vLLM 启动脚本
├── judge_rubrics.md             # per-task 评判标准 + reference trajectory
└── ALGORITHM_DESIGN.md          # 本文档
```

不修改 veRL 源码。通过 veRL 的 `@register` + `agent_loop_config_path` 机制注入自定义 agent loop。

---

## 9. 运行步骤

```bash
# 1. 阿里云 ECS 初始化
ssh root@8.163.82.224 'bash -s' < rl/scripts/setup_ecs.sh

# 2. RunPod 上拉取代码
cd /workspace/pinchbench-skill && git pull

# 3. 准备 prompt 数据
python rl/train/prepare_prompts.py --tasks-dir tasks/ --output-dir rl/data/prompts/

# 4. 设置环境变量
export OPENCLAW_HOST=8.163.82.224
export OPENCLAW_USER=root
export JUDGE_API_KEY=sk-xxx          # DashScope qwen-plus
export REWARD_MODE=self-judge        # 默认: self-judge

# 5. 启动训练
bash rl/train/run_reinforce_lora.sh

# 6. 运行 ablation (改 REWARD_MODE 即可)
REWARD_MODE=baseline bash rl/train/run_reinforce_lora.sh
REWARD_MODE=rule bash rl/train/run_reinforce_lora.sh
```

---

## 10. 关键设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| Online vs Offline RL | Online | 离线数据只能用一轮，live-user 场景必须 online |
| rollout.n | 1 | Live-user 场景，没有重试机会 |
| Terminal reward | {-1, +1} | 自带方向信号，不需要 baseline |
| Process reward | [-0.5, +0.3] | 量级够大，不被 terminal 淹没 |
| PRM 执行者 | Qwen3-4B (self-judge) | 自进化，随训练变强 |
| 天眼来源 | qwen-plus 成功轨迹 | 写入 rubric，给 self-judge 参考 |
| Advantage | 直接用 reward | process + terminal 都自带方向 |
| Batch size | 8 | 每个 task 1 条，不跨 task 做 baseline |
| 微调方式 | LoRA (rank=32) | 单卡 L4 可跑 |
| veRL 改动 | 零 | 全部通过外部注册 |
