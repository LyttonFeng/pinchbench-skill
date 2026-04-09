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
| ~~Baseline 测试~~ | ✅ 完成 | Qwen3-4B 33.7%, Qwen-turbo 34.2% |
| ~~选择训练 Task~~ | ✅ 完成 | 8 个 task，覆盖 6 个 category |
| **数据采集** | 🔄 进行中 | 8 task × N 次采样 |
| 数据转换 | ⏳ 待开始 | rescore + prepare_data → parquet |
| veRL 训练 | ⏳ 待开始 | GPG + process reward |
| 评测新 checkpoint | ⏳ 待开始 | PinchBench 复测 |
| 写 report | ⏳ 待开始 | |

---

## 训练目标（8 个 task）

| Task | Category | Qwen3-4B | Qwen-turbo | Grading |
|------|----------|----------|------------|---------|
| `task_02_stock` | research | 0% | 100% | automated |
| `task_12_skill_search` | file_ops | 0% | 67% | automated |
| `task_10_workflow` | complex | 33% | 33% | hybrid |
| `task_22_second_brain` | memory | 50% | 50% | hybrid |
| `task_16_email_triage` | organization | 38% | 32% | hybrid |
| `task_19_spreadsheet` | data_analysis | 20% | 20% | hybrid |
| `task_18_market_research` | research | 44% | 44% | hybrid |
| `task_24_polymarket` | research | 21% | 25% | hybrid |

---

## 训练 Pipeline

```
collect.py          → samples_raw.jsonl      （openclaw + Qwen3-4B 采样）
rescore.py          → samples_rescored.jsonl  （vLLM 补 logprobs）
prepare_data.py     → verl/train.parquet      （转 veRL 格式）
run_verl.sh         → rl/checkpoints/         （veRL GPG 训练）
benchmark.py        → results/               （评测新 checkpoint）
```

**算法**：veRL GPG（single-sample PG）+ KL 正则 + process reward
**Reward**：immediate（幻觉/空回复）+ next-state（tool 执行结果）+ terminal（grading 分数）
**Reward Manager**：自定义 veRL RewardManager，per-turn 分配 reward 到 `<|im_end|>` token
**数据 Schema**：`rl/train/data.py`（`TrainingSample`）

---

## 关键配置

| 项 | 值 |
|---|---|
| 推理服务 | RunPod L4 24GB + vLLM 0.19.0 |
| vLLM 参数 | `--tool-call-parser hermes --reasoning-parser deepseek_r1` |
| 本地连接 | SSH 隧道 `localhost:8000 → RunPod:8000` |
| OpenClaw 模式 | `--local`（绕过 gateway） |
| Judge 模型 | `dashscope/qwen-plus` |

---

## 交付物

1. ✅ Qwen3-4B 未训练 baseline 分数（33.7%）
2. RL 训练后分数（8 个 task reward 曲线）
3. 完整训练脚本（`rl/train/` 目录）
4. 1 页 report

---

## 后续

- Phase 2：扩到更多 task，换独立 judge，消除自评偏差
- Phase 3：迁移到 JiuwenClaw 真实 runtime，Online RL
