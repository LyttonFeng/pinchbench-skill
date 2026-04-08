# PinchBench RL Roadmap

## 背景

用 **PinchBench** 模拟 Live-User 场景，训练小模型（Qwen3-1.7B）学会用工具完成真实用户任务。

**为什么用 PinchBench 而不是 claw-eval？**

claw-eval 的评测任务在 mock 环境下运行，agent 拿到的上下文是预设好的，不是真实 runtime。模型在 claw-eval 上学到的策略未必能迁移到真实用户场景——评测指标好看，但不代表用户真的受益。

PinchBench 的 task 是真实用户请求类型，通过 openclaw agent 在真实 runtime 环境里执行，reward 来自确定性 grading 函数，没有人工干预。这个设定和 Live-User 场景在结构上一致。

**计划**：先用 PinchBench 把整个 RL pipeline 跑通（采样 → reward → 训练 → 评测），验证算法可行性；之后平移到 JiuwenClaw runtime，pipeline 不需要大改。

---

**Baseline**：qwen-plus 在 PinchBench 上得分 **81.3%**（25 个 task，judge 也是 qwen-plus 自评）。
目标：Qwen3-1.7B 经 RL 训练后在 hard bucket 任务上超过 SFT baseline。

---

## Phase 0：环境验证（已完成）

| 状态 | 内容 |
|------|------|
| ✅ | PinchBench 本地跑通（openclaw + DashScope qwen-plus） |
| ✅ | 自定义 endpoint / judge 支持 |
| ✅ | 跑完 25 个 task，得到 baseline 分数 81.3% |
| ✅ | `analyze.py` 分析 hard / very_hard / easy 分布 |
| ✅ | RL 数据格式（`TrainingSample`）定义完成 |
| ✅ | 三层 reward 实现（immediate + next-state + terminal） |
| ✅ | `collect.py` 支持多次采样（Live-User 循环） |
| ✅ | `rescore.py` vLLM 补 logprobs |
| ✅ | `train.py` single-sample PG 训练框架 |

---

## Phase 1：冷启动实验（本周）

**目标**：跑通完整 pipeline，得到第一个 RL checkpoint，出对比数字。

| 步骤 | 内容 |
|------|------|
| 1.1 | 租 RunPod L4/A10G，起 vLLM serve（Qwen3-1.7B） |
| 1.2 | 跑 Qwen3-1.7B **未训练 baseline**（直接跑 PinchBench 出分） |
| 1.3 | 对 3 个 hard task 各采样 30 次（`collect.py --runs 30`） |
| | `task_18_spreadsheet_summary`（数字汇总，automated，当前 0.60） |
| | `task_03_blog`（写作长度，llm_judge，当前 0.61） |
| | `task_24_polymarket_briefing`（内容完整性，hybrid，当前 0.67） |
| 1.4 | `rescore.py` 补 logprobs |
| 1.5 | `train.py` RL 训练（single-sample PG + LoRA，3 epoch） |
| 1.6 | PinchBench 跑 RL checkpoint，对比 baseline |
| 1.7 | 写 report |

**交付物**：

1. Qwen3-1.7B baseline 分数（vs qwen-plus 81.3%）
2. RL 训练后分数（vs baseline，按 category 拆分）
3. Report：方法 + 数字 + 结论 + 下一步

---

## Phase 2：扩大训练集 + 独立 Judge（下周）

**目标**：提升数据质量和评测可信度。

| 步骤 | 内容 |
|------|------|
| 2.1 | 给所有 12 个 hard task 各采样 30 次，共 ~360 条 trajectory |
| 2.2 | 换独立 judge 模型（Qwen3-8B 或 claude），消除 qwen-plus 自评偏差 |
| 2.3 | 对比 SFT baseline vs RL，分析 trajectory 质量变化 |
| 2.4 | 分析 reward 三层分布（immediate / next-state / terminal 各贡献多少） |

---

## Phase 3：迁移到 JiuwenClaw + Online RL

**目标**：从 PinchBench sandbox 迁移到真实 JiuwenClaw runtime，实现真正的 Live-User Online Update。

| 步骤 | 内容 |
|------|------|
| 3.1 | 把 openclaw 替换为 JiuwenClaw runtime |
| 3.2 | collect → rescore → train 自动化串联（pipeline 脚本） |
| 3.3 | 每轮训练后用 val split 评估，防止过拟合 |
| 3.4 | checkpoint 发布，PinchBench 官方榜单提交 |

---

## RL 算法概述

**Single-sample PG**（适合 Live-User 场景，不用 GRPO）：

```
用户请求 → openclaw + Qwen3-1.7B 执行一次
         → transcript（真实 tool call 序列）
         → 三层 reward 打分
         → vLLM re-score 补 logprobs
         → turn-level clipped PG 更新
```

**三层 Reward**（全规则，无 LLM judge 介入）：

| 层 | 触发条件 | 分值 |
|----|---------|------|
| immediate | 幻觉 / 空回复 / 拒绝执行 | -1 |
| next-state | tool 执行出错 / 空返回 / 正常 | -0.5 / -0.1 / +0.2 |
| terminal | grading 函数输出，广播到所有 turn | [0, 1] |

**训练 Loss**：

```
L = L_PG + β_kl * KL(π_θ || π_ref) + c_critic * L_critic
```

---

## 关键风险

| 风险 | 说明 | 缓解 |
|------|------|------|
| judge 自评偏差 | qwen-plus 自评分虚高，llm_judge task 信号有噪声 | Phase 1 只用 automated task；Phase 2 换独立 judge |
| Qwen3-1.7B 能力上限 | 模型太小，某些任务可能无法完成 | 先做 automated task，信号干净，能力边界清晰 |
| trajectory 截断 | openclaw 有 timeout，长任务可能中途停 | `collect.py` 记录 `timed_out`，训练时过滤 |

---

## 指标

| 指标 | 当前（qwen-plus baseline） | Phase 1 目标 |
|------|--------------------------|-------------|
| PinchBench 总分 | 81.3% | Qwen3-1.7B RL 后超过 SFT baseline |
| Hard bucket 平均分 | 86.0% | RL 后 hard bucket reward 均值上升 |
| Automated task 平均分 | 89.0% | 保持或提升 |
