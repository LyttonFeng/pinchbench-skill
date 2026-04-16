# PinchBench RL 实验总结

**日期：** 2026-04-16  
**模型：** Qwen/Qwen3-4B + LoRA（REINFORCE++）  
**任务集：** RL8（8个典型 agent 任务）

---

## RL8 任务集概览

数据来源：run 0057（Qwen3-4B baseline，总分 50.4%）的真实 transcript 统计。


| 任务                          | 名称                                 | 核心能力                   | 主要工具/Skills                                        | 实际 turns | 实际 tool calls | 工作区文件                                          |
| --------------------------- | ---------------------------------- | ---------------------- | -------------------------------------------------- | -------- | ------------- | ---------------------------------------------- |
| task_02_stock               | Stock Price Research               | 实时信息检索 + 文件写入          | `web_search`, `write`                              | 5        | 4             | 无                                              |
| task_10_workflow            | Multi-step API Workflow            | 文件读取 + 代码生成 + 文档撰写     | `read`, `write`                                    | 3        | 3             | `config.json`                                  |
| task_12_skill_search        | Search and Replace in Files        | 多文件搜索替换 + 格式保留         | `read`, `write`（JSON/YAML）                         | **16**   | **49**        | `config/settings.json`, `config/database.yml`  |
| task_16_email_triage        | Email Inbox Triage                 | 批量阅读 + 优先级判断 + 结构化输出   | `read`（×13封）, `write`                              | 3        | 14            | 13封邮件 `inbox/email_01~13.txt`                  |
| task_18_market_research     | Competitive Market Research        | 网络搜索 + 分析报告撰写          | `web_search`, `web_fetch`, `write`                 | 3        | 4             | 无                                              |
| task_18_spreadsheet_summary | CSV & Excel Data Summarization     | 多格式数据解析 + 数值计算 + 汇总    | `read`（CSV/XLSX）, `execute`, `write`               | 3        | 3             | `quarterly_sales.csv`, `company_expenses.xlsx` |
| task_22_second_brain        | Second Brain Knowledge Persistence | 文件持久化 + 跨 session 记忆召回 | `mkdir`, `write`, `read`                           | 4        | 1             | 3 sessions                                     |
| task_24_polymarket_briefing | Polymarket + News Briefing         | 实时 API 抓取 + 新闻检索 + 反幻觉 | `web_fetch`（Polymarket API）, `web_search`, `write` | 9        | 8             | 无                                              |


**说明：**

- task_12 异常高（16 turns / 49 tool calls）：baseline 在 search-replace 上大量反复试错，得分 0%，是整个 RL8 里最难的任务
- task_12 名为 "skill_search" 是历史命名，实际是配置文件搜索替换操作
- task_18 有两个变体：`market_research`（纯研究写作）和 `spreadsheet_summary`（数据分析），id 前缀相同但任务性质差异大
- task_22 是唯一的多 session 任务，turns 为 3 sessions 合计

---

## 一、实验背景

PinchBench-RL8 从 PinchBench 中筛选了 8 个典型任务，覆盖工具调用、文件操作、信息检索、结构化分析、记忆持久化和多轮任务推进等场景。选取标准是：`qwen-plus` 在这些任务上显著强于 `Qwen3-4B` base，有区分度，适合验证在线 RL 的有效性。

---

## 二、算法设计：相对标准 REINFORCE++ 的三项改进

标准 REINFORCE++ 针对 live-user agent 场景有三个核心问题：任务异质性导致 batch-level baseline 统计不准、多轮 episode 的 reward 稀疏性、以及通用 judge 对任务特定行为的评分不准。我们针对这三点做了改进。

### 改进 1：Per-Task EMA Baseline

**问题：** batch 里混有始终 0 分的任务（task_12）和始终满分的任务（task_22），batch-level whitening 把 baseline 拉到中间，导致已收敛任务的 advantage 接近 0，失败任务的 advantage 持续为负但没有正确方向。

**方案：** 每个任务维护独立的 EMA baseline，advantage = raw_reward - task_EMA。始终失败的任务 advantage 趋近 0（不产生无效梯度），新学会的任务 advantage 为强正信号，已收敛的任务自动停止更新。

### 改进 2：Turn-Level Dense Reward

**问题：** episode 级 scalar reward 在 10~15 轮的 agent 轨迹上 credit assignment 极难，模型无法区分哪一步工具调用导致了最终结果。

**方案：** 每个 assistant turn 独立打分（PRM），terminal reward 叠加在最后一个 turn 上。两级信号分工：PRM 负分惩罚明显错误行为（fabrication、重复失败命令），terminal 正分奖励任务完成。

### 改进 3：Rubric-Guided Oracle Judge

**问题：** 通用 LLM judge 对 agent 行为的评分方差高，无法区分任务特定的正确/错误模式。

**方案：** 为每个任务设计结构化 rubric（goal + optional_hints + common_mistakes），oracle judge（qwen-plus）结合 rubric 对每个 turn 打分。common_mistakes 直接对应训练中观察到的失败模式，形成"trajectory analysis → rubric fix → retrain"的闭环。Self-judge 变体（Mode C）用训练中的 Qwen3-4B 自身打分，实现自进化，无需外部 API。

---

## 三、实验结果


| 任务                          | 任务描述                                      | 典型 step 数 | qwen-plus（上限） | Qwen3-4B baseline | Qwen3-4B + LoRA（2-step-only） | 相对 baseline（pp） |
| --------------------------- | ----------------------------------------- | ---------- | ------------- | ----------------- | ----------------------------- | ---------------- |
| task_02_stock               | 股价调研：检索行情并写入文件                            | 5          | 100%          | 67%               | 92%                           | **+25.0**        |
| task_10_workflow            | 多步 API 工作流：读配置、生成代码/脚本、写文档                 | 3          | 87.9%         | 33%               | 77%                           | **+44.0**        |
| task_12_skill_search        | 多文件配置搜索替换（JSON/YAML），易触发长轮试错                 | 16         | 100%          | 0%                | 17%                           | **+17.0**        |
| task_16_email_triage        | 邮件 triage：批量阅读、优先级分类、结构化输出                 | 3          | 89.1%         | 39%               | 89%                           | **+50.0**        |
| task_18_market_research     | 竞品市场研究：检索 + 抓取 + 报告撰写                      | 3          | 88.0%         | 34%               | 79%                           | **+45.0**        |
| task_18_spreadsheet_summary | CSV/XLSX 数据解析、计算与汇总输出                       | 3          | 97.5%         | 20%               | 2.5%                          | **-17.5**        |
| task_22_second_brain        | Second brain：跨 session 持久化与召回               | 4          | 100%          | 0%                | 100%                          | **+100.0**       |
| task_24_polymarket_briefing | Polymarket + 新闻简报：API、搜索、反幻觉                 | 9          | 58.3%         | 12%               | 54%                           | **+42.0**        |
| **总分**                      | —                                         | —          | **90.1%**     | **50.4%**         | **66.0%**                     | **+15.6**        |

「典型 step 步数」与上文 **RL8 任务集概览** 中 run 0057 baseline 的 **实际 turns**（一次跑下来的 assistant 轮次深度）一致，便于对照任务难度与分数变化。

**表注（v1 / 2-step-only）：**

- **总分绝对提升：** Qwen3-4B baseline **50.4%** → 上表 LoRA（v1 **step 8** 检查点）**66.0%**，**+15.6 个百分点（pp）**。
- **训练强度：** RL8 共 8 条样本；配置为 `BATCH_SIZE=2`、`TOTAL_EPOCHS=3`，每条轨迹在训练循环中**仅被优化约 2 次（2-step-only）** 即达到该列分数；step 8 为最佳检查点（后续 step 出现过训练退化，见下文 v1 描述）。

---

## 四、v1 → v2 → v3 演进

### v1（已完成）

REINFORCE++ + LoRA，scalar episode reward，self-judge，无 KL 惩罚。step 8 达到 66%（+15.6 pp），step 17 退化到 51%（过训练）。

**重要观察：** 每个样本仅学习 2 次（BATCH=2，TOTAL_EPOCHS=3，8 tasks），训练极轻，step 8 即达到最佳。

### v2（已完成，step6=46.3%）

引入 turn-level reward、oracle-judge、KL 惩罚、BATCH=4。工程侧修复了 shape crash，启用 LORA_ONLY_CKPT。

**退化根因：** 4 个任务的 rubric 存在质量问题——描述了 OpenClaw 里不存在的工具行为（task_18 引用 awk/exec），或未覆盖关键失败模式（task_02 拒绝写文件、task_24 日期幻觉、task_12 glob 路径错误）。Oracle judge 按错误 rubric 打分，模型学到了错误策略。

**核心教训：rubric 质量 > 算法超参。** 错误的 rubric 比没有 PRM 更差。

### v3（训练中）

在 v2 基础上修复 4 个任务的 rubric，同时调整奖励权重：


| 参数                     | v2      | v3                   |
| ---------------------- | ------- | -------------------- |
| TERMINAL_REWARD_WEIGHT | 0.5     | **0.7**（terminal 主导） |
| PRM 正向上限               | +0.3    | **+0.2**（降低虚假正奖励）    |
| TOTAL_TRAINING_STEPS   | 8       | **8**                |
| TRAINER_RESUME_MODE    | disable | disable              |


Rubric 修复要点：

- **task_02**：拒绝写文件 = 最差结果；有残缺数据也要写文件
- **task_12**：禁止 glob 路径；ENOENT 后不能放弃
- **task_18**：xlsx 返回 binary 是正常的，用 prompt 结构估算；删除不可能的 awk 引用
- **task_24**：禁止写错年份；禁止回退 2023 训练数据；至少搜 3-4 次

---

## 五、当前状态

- v3 训练已于 2026-04-16 在 RunPod 启动（tmux: `train_v3`，checkpoint: `reinforce_lora_v3/`）
- v2 step6 LoRA 已备份至 `rl/checkpoints/reinforce_lora_v2_step6_lora_backup`
- 每 2 steps 保存一次，step 2/4/6/8 自动跑 val
- 成功标准：val-core 超过 v1 的 66%，且无退化趋势

---

## 六、一句话总结

> 三版实验验证了 REINFORCE++ + LoRA 在 live-user agent 场景的有效性（v1 +15.6 pp），并逐步定位到 rubric 质量是 PRM 训练的核心瓶颈；v3 在修复 rubric 的同时加强 terminal 信号权重，目标是在 8 steps 内稳定超越 v1。

