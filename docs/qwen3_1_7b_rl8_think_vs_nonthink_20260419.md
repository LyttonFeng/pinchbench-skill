# Qwen3-1.7B RL8 对照结果

日期：2026-04-19

## 结论

同一套 RL8 benchmark 下，`Qwen/Qwen3-1.7B` 的 `think mode` 明显优于 `non-think`：

- `non-think`: `34.5%`
- `think`: `53.0%`

这说明：

- 对部分多步任务，`<think>` 能明显改善轨迹质量和最终分数。
- 但它不是对所有任务都有效，`task_12_skill_search` 在两种模式下都还是 `0`。
- `think` 可能会让某些解析/数据分析任务更容易走偏，`task_18_spreadsheet_summary` 在 think 下反而更差。

## 运行配置

### Non-think

- 模型：`Qwen/Qwen3-1.7B`
- 开关：`OPENCLAW_MODEL_REASONING=0`
- RL8 总分：`2.76/8`
- 得分：`34.5%`

### Think

- 模型：`Qwen/Qwen3-1.7B`
- 开关：`OPENCLAW_MODEL_REASONING=1`
- RL8 总分：`4.24/8`
- 得分：`53.0%`

## 单项对比

| Task | Non-think | Think | 变化 |
|------|-----------|-------|------|
| `task_02_stock` | `0.8333` | `0.9000` | 上升 |
| `task_10_workflow` | `0.4583` | `0.5000` | 上升 |
| `task_12_skill_search` | `0.0000` | `0.0000` | 不变 |
| `task_16_email_triage` | `0.5114` | `0.7273` | 明显上升 |
| `task_18_market_research` | `0.6300` | `0.6100` | 小幅下降 |
| `task_18_spreadsheet_summary` | `0.3250` | `0.1750` | 下降 |
| `task_22_second_brain` | `0.0000` | `1.0000` | 大幅上升 |
| `task_24_polymarket_briefing` | `0.0000` | `0.3333` | 上升 |

## 观察

### 1. think 对多步记忆类任务帮助最大

`task_22_second_brain` 从 `0` 到 `1.0`，是最明显的收益。说明 `think mode` 更利于跨会话记忆保存、回忆和复用。

### 2. think 对邮箱分流类任务也有帮助

`task_16_email_triage` 提升明显，说明对优先级判断、上下文联系、长列表整理这类任务，think 能减少粗糙决策。

### 3. think 不是万能药

`task_12_skill_search` 仍然是 `0`，说明这个任务不是单纯靠更长的思考链条就能解决，更像是工具使用、路径发现或者执行链路的问题。

### 4. think 可能拖累数据解析类任务

`task_18_spreadsheet_summary` 在 think 下更差，说明当任务本身需要快速、结构化的数据读取时，额外思考不一定带来收益，甚至可能引入更多错误路径。

## 结果文件

- non-think: `results/0117_qwen3-1-7b.json`
- think: `results/0118_qwen3-1-7b.json`

## 备注

这两轮结果都来自同一套 RL8 suite，区别仅在于是否开启 `OPENCLAW_MODEL_REASONING`。
因此这组数据可以作为后续训练和推理配置讨论的对照基线。
