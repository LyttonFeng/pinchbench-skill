# PinchBench RL — 项目计划

用 PinchBench 模拟 Live-User 场景，训练 Qwen3-4B 学会用工具完成任务。

## 为什么不用 claw-eval

claw-eval 是 mock 环境，PinchBench 是真实 runtime，结构和 Live-User 一致，pipeline 跑通后直接平移到 JiuwenClaw。

## Baseline（已完成 ✅）

| 模型 | 总分 | 说明 |
|------|------|------|
| qwen-plus | 81.3% | 大模型上限参考 |
| qwen-turbo (DashScope) | 34.2% | 中等模型参考 |
| **Qwen3-4B (vLLM)** | **33.7%** | **RL 训练前 baseline** |

---

## 进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| ~~环境搭建~~ | ✅ 完成 | RunPod L4 + vLLM + SSH 隧道 + OpenClaw --local |
| ~~Baseline 测试~~ | ✅ 完成 | Qwen3-4B 33.7%, Qwen-turbo 34.2%, qwen-plus 81.3% |
| ~~选择训练 Task~~ | ✅ 完成 | 8 个 task，覆盖 6 个 category |
| ~~算法设计~~ | ✅ 完成 | Online RL + GRPO + LoRA + self-judge PRM |
| ~~训练架构~~ | ✅ 完成 | RunPod(训练+vLLM) ↔ ECS(OpenClaw) ↔ DashScope(judge) |
| ~~代码开发~~ | ✅ 完成 | agent_loop, model_proxy, trajectory, reward, 训练脚本 |
| ~~Judge Rubric~~ | ✅ 完成 | 8 个 task 的天眼 reference trajectory + per-turn 评判标准 |
| ~~离线数据采集~~ | ❌ 废弃 | 改为 Online RL，不需要预采集 |
| **ECS 环境初始化** | ⏳ 待开始 | 阿里云 ECS 安装 OpenClaw + PinchBench |
| **端到端联调** | ⏳ 待开始 | RunPod ↔ ECS 跑通一个完整 episode |
| **训练实验** | ⏳ 待开始 | 4 组 ablation (baseline/rule/self-judge/oracle-judge) |
| **评测新 checkpoint** | ⏳ 待开始 | PinchBench 全量复测 |
| **写 report** | ⏳ 待开始 | 实验结果 + ablation 分析 |

---

## 算法设计（已确定）

### 核心方案：Online RL + Self-Judge PRM

```
每个 training step:
  1. 从 8 个 task 采样 batch_size=8
  2. 用当前 LoRA 策略跑 OpenClaw episode
  3. PinchBench grading → terminal reward {-1, +1}
  4. Qwen3-4B self-judge → per-turn process reward [-0.5, +0.3]
  5. GRPO update (rollout.n=1 退化为 REINFORCE)
```

### 为什么不用离线 RL

- GRPO 需要 re-rollout，离线数据只能用一轮
- rollout.n=1 没有 baseline，单靠 terminal reward 学不动
- **Online RL + process reward 解决了这两个问题**

### Reward 设计

| 信号 | 范围 | 来源 | 说明 |
|------|------|------|------|
| Terminal | {-1, +1} | PinchBench grading (qwen-plus judge) | 自带方向，不需要 baseline |
| Process | [-0.5, +0.3] | Qwen3-4B self-judge (天眼 rubric) | 独立于 terminal，确保 0% task 也能学习 |

### Self-Judge 自进化

- Agent 和 Judge 是同一个 Qwen3-4B
- Judge 有额外信息：rubric + 天眼 reference trajectory
- 类比：学生考试不会做，但给标准答案让他改卷，他是能改的
- Agent 变强 → Judge 也变强 → 正循环

### 4 组 Ablation 实验

| 实验组 | PRM 方式 | 天眼 | 说明 |
|-------|---------|:---:|------|
| A: Baseline | 无 | ✗ | 纯 terminal reward |
| B: Rule-only | Python 规则 | ✗ | 无 LLM 调用 |
| **C: Self-Judge** | **Qwen3-4B** | **✓** | **默认：自进化** |
| D: Oracle-Judge | qwen-plus API | ✓ | 对照组 |

---

## 训练目标（8 个 task）

| Task | Category | Qwen3-4B | qwen-plus | Grading |
|------|----------|:--------:|:---------:|---------|
| `task_02_stock` | research | 0% | 100% | automated |
| `task_10_workflow` | complex | 33% | 75% | hybrid |
| `task_12_skill_search` | file_ops | 0% | 100% | automated |
| `task_16_email_triage` | organization | 38% | 50% | hybrid |
| `task_18_market_research` | research | 44% | 100% | hybrid |
| `task_19_spreadsheet` | data_analysis | 20% | 50% | hybrid |
| `task_22_second_brain` | memory | 50% | 75% | hybrid |
| `task_24_polymarket` | research | 21% | 100% | hybrid |

---

## 训练架构

```
┌── RunPod (L4 GPU) ─────────────┐        ┌── 阿里云 ECS (4核 8G) ──┐
│                                 │        │                          │
│  veRL Trainer                   │  SSH   │  OpenClaw (--local)      │
│    ├── Qwen3-4B + LoRA (r=32)  │ ─────► │    ├── 工具执行          │
│    ├── vLLM (Agent + PRM 共享) │        │    └── models.json       │
│    ├── Ref Model (CPU)         │ ◄───── │         → ModelProxy     │
│    └── ModelProxy (:PORT)      │  HTTP  │                          │
│                                 │        │  PinchBench Grading      │
│  显存: vLLM 45% + Actor 55%   │        │    └── qwen-plus API     │
└─────────────────────────────────┘        └──────────────────────────┘
```

- 不修改 veRL 源码，通过 `@register` 机制注入
- OpenClaw 在 ECS 7x24 运行，不受 GPU 实例启停影响
- 估算: ~8-12 分钟/step, 400 episodes 约 6-10 小时

---

## 训练 Pipeline

```
prepare_prompts.py  → rl/data/prompts/train.parquet   （8 个 task prompt）
setup_ecs.sh        → 阿里云 ECS 环境初始化            （OpenClaw + PinchBench）
run_grpo_lora.sh    → rl/checkpoints/grpo_lora/        （Online RL 训练）
benchmark.py        → results/                         （评测新 checkpoint）
```

### 训练超参

| 参数 | 值 | 理由 |
|------|-----|------|
| LoRA rank | 32 | 平衡参数量和表达力 |
| Learning rate | 2e-5 | LoRA 常用值 |
| Batch size | 8 | 每个 task 1 条 |
| KL coef | 0.05 | 防止偏离 ref 太远 |
| Temperature | 0.7 | rollout 探索温度 |
| GPU memory (vLLM) | 45% | 剩余给 Actor training |
| Total epochs | 50 | 50 × 8 = 400 episodes |

---

## 关键配置

| 项 | 值 |
|---|---|
| GPU | RunPod L4 24GB |
| OpenClaw 主机 | 阿里云 ECS 8.163.82.224 |
| 推理引擎 | vLLM (hermes parser + deepseek_r1 reasoning parser) |
| Agent 模型 | Qwen3-4B + LoRA |
| PRM 模型 | Qwen3-4B (self-judge, 同一个 vLLM) |
| Terminal Judge | qwen-plus (DashScope API) |
| 天眼来源 | qwen-plus 成功轨迹，写在 judge_rubrics.md |

---

## 代码结构

```
rl/
├── agent_loop/
│   ├── openclaw_agent_loop.py   # 核心: 驱动 OpenClaw + ModelProxy
│   ├── model_proxy.py           # HTTP 反向代理
│   ├── trajectory.py            # token-level 轨迹重建
│   ├── reward.py                # PRM: self-judge + rubric
│   └── config.yaml              # veRL 注册配置
├── train/
│   ├── run_grpo_lora.sh         # 训练启动脚本
│   ├── prepare_prompts.py       # 数据准备
│   └── reward_manager.py        # veRL reward 适配
├── scripts/
│   ├── setup_ecs.sh             # ECS 初始化
│   └── start_vllm.sh            # vLLM 启动
├── judge_rubrics.md             # 评判标准 + 天眼 reference
├── ALGORITHM_DESIGN.md          # 算法设计文档
└── TRAINING_ARCHITECTURE.md     # 训练架构文档
```

---

## 交付物

1. ✅ Qwen3-4B 未训练 baseline 分数（33.7%）
2. ✅ 算法设计文档 (`rl/ALGORITHM_DESIGN.md`)
3. ✅ 训练架构文档 (`rl/TRAINING_ARCHITECTURE.md`)
4. ✅ 完整训练代码 (`rl/agent_loop/` + `rl/train/`)
5. ✅ Judge Rubric + 天眼 reference (`rl/judge_rubrics.md`)
6. ⏳ 4 组 ablation 实验结果 + reward 曲线
7. ⏳ RL 训练后 PinchBench 复测分数
8. ⏳ 实验报告

---

## 后续

- Phase 2：扩到更多 task，验证 self-judge 泛化性
- Phase 3：迁移到 JiuwenClaw 真实 runtime，真正 Live-User Online RL
- 论文方向：Self-Judge PRM 在 agentic RL 中的有效性
