# PinchBench RL 实验总结（会议版）

**日期：** 2026-04-16
**模型：** Qwen/Qwen3-4B + LoRA（REINFORCE++）
**任务集：** RL8（8个典型 agent 任务）

---

## 一、实验背景

PinchBench-RL8 从 PinchBench 中筛选了 8 个典型任务，覆盖工具调用、文件操作、信息检索、结构化分析、记忆持久化和多轮任务推进等场景。选取标准是：`qwen-plus` 在这些任务上显著强于 `Qwen3-4B` base，有区分度，适合验证在线 RL 的有效性。

---

## 二、v1 实验：已完成，step 8 最佳

### 算法设计

| 组件 | 设计 |
|---|---|
| 训练范式 | REINFORCE++ + LoRA（在线 rollout，不依赖 critic） |
| 基座模型 | Qwen/Qwen3-4B |
| Reward 构成 | process reward（self-judge，按 rubric 逐 turn 打分）+ terminal reward（PinchBench grading） |
| Reward 返回方式 | scalar（episode 级汇总） |
| KL 惩罚 | 无（use_kl_in_reward=False） |
| 多轮 runtime | OpenClaw + vLLM，训练与推理 runtime 统一 |

### v1 关键超参

| 参数 | 值 |
|---|---|
| BATCH_SIZE | 2 |
| TOTAL_EPOCHS | 3 |
| TEST_FREQ / SAVE_FREQ | 4 / 4（每 epoch 一次） |
| MAX_TURNS | 16 |
| MAX_PROMPT_LENGTH | 20000 |
| MAX_RESPONSE_LENGTH | 12000 |
| VLLM_GPU_MEM_UTIL | 0.28 |
| LORA_RANK / ALPHA | 32 / 64 |
| LR | 2e-5 |
| TERMINAL_REWARD_WEIGHT | 0.3 |
| TASK_EMA_INIT | 0.5 |
| TRAINER_RESUME_MODE | disable |

### v1 实验结果

| 任务 | qwen-plus（上限） | Qwen3-4B baseline | v1 LoRA step8 |
|---|---:|---:|---:|
| task_02_stock | 100% | 67% | 92% |
| task_10_workflow | 87.9% | 33% | 77% |
| task_12_skill_search | 100% | 0% | 17% |
| task_16_email_triage | 89.1% | 39% | 89% |
| task_18_market_research | 88.0% | 34% | 79% |
| task_18_spreadsheet_summary | 97.5% | 20% | 2.5% |
| task_22_second_brain | 100% | 0% | 100% |
| task_24_polymarket_briefing | 58.3% | 12% | 54% |
| **总分** | **90.1%** | **50.4%** | **66.0%** |

> **绝对提升 +15.6 pp**（50.4% → 66.0%）

**重要观察：** v1 在 step 17 退化到 51.6%，与 baseline 持平，说明出现了**过训练**。step 8 是最佳检查点。

---

## 三、v1 实验缺陷

### 1. 过训练问题（核心问题）
step 8 → step 17 分数从 66% 跌回 ~51%。原因：
- BATCH_SIZE=2 导致梯度信号方差大，容易在单任务上反复强化错误行为
- 无 KL 惩罚，模型可以无约束偏离 reference policy
- TASK_EMA_INIT=0.5 偏高，early stage advantage 全为负，信号不稳定

### 2. task_18_spreadsheet_summary 明显退化（20% → 2.5%）
模型没有学会对 `.xlsx` 进行结构化解析，把 Excel 文件当作二进制文本读取，导致输出完全错误。原因：self-judge rubric 没有覆盖"必须先正确解析文件格式"这一前置条件。

### 3. task_12_skill_search 几乎没有改善（0% → 17%）
多文件修改场景下，模型的工具调用路径仍不稳定。

### 4. Reward 信号稀疏（scalar 模式）
v1 使用 episode 级汇总 reward，reward 被分摊到所有 token，每个 turn 的行为信号非常弱。对多轮任务（16 turns）来说，梯度无法有效区分好/坏的中间步骤。

### 5. 工程稳定性问题
- BATCH_SIZE=2 太小，checkpoint 体积大（FSDP 全量权重 ~17GB），磁盘压力大
- 当时未启用 LORA_ONLY_CKPT，导致存储开销高

---

## 四、v2 实验：当前进行中（已启动）

### v1 → v2 核心改动

| 改动点 | v1 | v2 | 理由 |
|---|---|---|---|
| BATCH_SIZE | 2 | **4** | 降低梯度方差，训练更稳定 |
| use_kl_in_reward | False | **True** | 防止模型漂离 reference policy，抑制过训练 |
| actor.use_kl_loss | False | **True**（coef=0.05） | 额外 loss 侧 KL 约束 |
| TASK_EMA_INIT | 0.5 | **0.3** | 与 terminal_reward_weight=0.5 对齐，避免 early stage 全负 advantage |
| TERMINAL_REWARD_WEIGHT | 0.3 | **0.5** | 加强任务完成的最终信号 |
| Reward 返回方式 | scalar | **turn-level** | 每个 turn 的行为都有独立梯度信号，显著降低稀疏性 |
| Reward 评判 | self-judge | **oracle-judge**（qwen-plus） | 评判质量更高，信号更准 |
| TOTAL_EPOCHS | 3 | **4** | 覆盖 8 个 task（4 batch/epoch × 4 epochs = 8 steps，防过训练） |
| LORA_ONLY_CKPT | 0 | **1** | 只保存 LoRA adapter（~50MB），大幅降低磁盘开销 |
| 代码修复 | — | **Turn-0 左截断** | 防止超长初始 prompt（如 task_16 的 17053 tokens）导致 veRL shape 不一致 crash |

### v2 关键超参

| 参数 | 值 |
|---|---|
| BATCH_SIZE | 4 |
| TOTAL_EPOCHS | 4（共 8 steps） |
| TEST_FREQ / SAVE_FREQ | 2 / 2 |
| MAX_TURNS | 16 |
| MAX_PROMPT_LENGTH | 20000 |
| MAX_RESPONSE_LENGTH | 12000 |
| VLLM_GPU_MEM_UTIL | **0.35** |
| VLLM_MAX_MODEL_LEN | **32768** |
| LORA_RANK / ALPHA | 32 / 64 |
| LR | 2e-5 |
| TERMINAL_REWARD_WEIGHT | **0.5** |
| TASK_EMA_INIT | **0.3** |
| KL loss coef（actor侧） | **0.05** |
| TRAINER_RESUME_MODE | disable |
| RUN_VERSION | v2 |

### v2 算法流程

```
用户 task
  → OpenClaw + vLLM rollout（最多 16 turns）
  → 每 turn：oracle-judge（qwen-plus）按 rubric 打分 → turn-level reward
  → 终局：PinchBench grading → terminal reward（weight=0.5）
  → per-task EMA baseline 归一化 advantage
  → REINFORCE++ 更新 LoRA（+ KL 惩罚）
  → 循环
```

---

## 五、v1 vs v2 对比一览

| 维度 | v1 | v2 |
|---|---|---|
| 最佳结果 | 66.0%（step 8） | 进行中 |
| 过训练风险 | 高（step 17 跌回 51%） | 低（KL 惩罚 + BATCH_SIZE=4） |
| Reward 稀疏性 | 高（scalar episode） | 低（turn-level） |
| 评判质量 | self-judge | oracle-judge（qwen-plus） |
| 工程稳定性 | 磁盘压力大，偶发 crash | LORA_ONLY_CKPT，代码修复 shape bug |
| 训练效率 | BATCH=2，方差大 | BATCH=4，梯度更稳 |

---

## 六、当前状态与预期

- v2 训练已于 2026-04-16 在 RunPod 启动（tmux session: `v2train`）
- 每 2 steps 保存一次 checkpoint，预计 8 steps 后可以评测
- 监控指标：`reward_score`、`advantage` 曲线、val-core reward mean
- 成功标准：step 4–8 的 val 分数超过 v1 step 8 的 66%，且无明显退化趋势

**monitor 命令：**
```bash
ssh -p 15416 -i ~/.ssh/id_ed25519 root@216.81.248.115 "tail -f /workspace/v2train.log"
```

---

## 七、task_16 可视化 Demo 与后续专项（workflow）

仓库内有一份 **静态 HTML 汇报页**，用于展示 **task_16（email triage）** 上 Baseline vs RL LoRA 的能力对比（读全邮件、优先级 triage、可交付输出等），适合会议投屏或手机查看：

- 路径：`docs/task_16_email_triage_demo.html`（浏览器直接打开即可）

**后续计划（与 RL8 全量实验并行）：** 在 **task_16_email_triage** 与 **workflow 类任务**（RL8 中为 `task_10_workflow`）上做一次 **针对性强化**，目标是把这两条线上的 **PinchBench 终局分拉到 100%**，并配合上述 Demo（或扩展一版 workflow 对比页）做 **可演示、可复述** 的明显提升故事。

---

## 八、一句话总结

> v1 实验验证了 REINFORCE++ + LoRA 在线 RL 的有效性（+15.6 pp），但过训练和稀疏 reward 是主要短板；v2 在算法侧引入 turn-level reward、KL 惩罚、oracle-judge，在工程侧修复了 shape crash，目标是在 8 steps 内达到更高峰值且保持稳定。
