# PinchBench Baseline 测试结果

## 测试环境

| 项 | 值 |
|---|---|
| 测试日期 | 2026-04-08 |
| PinchBench 版本 | pinchbench-skill (main branch) |
| Judge 模型 | dashscope/qwen-plus |
| Qwen3-4B 推理 | RunPod L4 24GB + vLLM 0.19.0 (`--tool-call-parser hermes --reasoning-parser deepseek_r1`) |
| Qwen-plus 推理 | DashScope API |

## 结果文件

| 模型 | 总分 | 结果文件 | Transcript 目录 |
|---|---|---|---|
| qwen-plus | 81.3% | `results/0004_qwen-plus.json` | `results/0004_transcripts/` |
| qwen-turbo | 34.2% | `results/0020_dashscope-qwen-turbo.json` | `results/0020_transcripts/` |
| Qwen3-4B | 33.7% | `results/0020_vllm-qwen3-4b.json` | `results/0020_transcripts/` |

---

## 一、Qwen3-4B vs qwen-plus 全量对比（25 task）

| Task | Category | Grading | qwen-plus | Qwen3-4B | 差距 |
|------|----------|---------|-----------|----------|------|
| task_00_sanity | basic | automated | 100% | 100% | — |
| task_01_calendar | calendar | automated | 100% | 83% | -17% |
| task_02_stock | research | automated | 100% | 0% | -100% |
| task_03_blog | writing | llm_judge | 61% | 0% | -61% |
| task_04_weather | research | automated | 100% | 79% | -21% |
| task_05_summary | writing | llm_judge | 98% | 0% | -98% |
| task_06_events | calendar | llm_judge | 92% | 0% | -92% |
| task_07_email | communication | llm_judge | 98% | 0% | -98% |
| task_08_memory | context | automated | 90% | 90% | 0% |
| task_09_files | file_ops | automated | 100% | 86% | -14% |
| task_10_workflow | complex | hybrid | 89% | 33% | -56% |
| task_11_clawdhub | complex | automated | 100% | 100% | 0% |
| task_12_skill_search | file_ops | automated | 100% | 0% | -100% |
| task_13_image_gen | creative | hybrid | 0% | 0% | 0% |
| task_14_humanizer | writing | llm_judge | 100% | 0% | -100% |
| task_15_daily_summary | writing | llm_judge | 98% | 0% | -98% |
| task_16_email_triage | organization | hybrid | 88% | 38% | -50% |
| task_17_email_search | communication | hybrid | 98% | 0% | -98% |
| task_18_market_research | research | hybrid | 93% | 44% | -49% |
| task_19_spreadsheet | data_analysis | hybrid | 60% | 20% | -40% |
| task_20_eli5_pdf | writing | llm_judge | 0% | 0% | 0% |
| task_21_openclaw | comprehension | automated | 0% | 0% | 0% |
| task_22_second_brain | memory | hybrid | 100% | 50% | -50% |
| task_24_polymarket | research | hybrid | 67% | 21% | -46% |
| task_25_access_log | data_analysis | automated | 100% | 100% | 0% |
| **总计** | | | **81.3%** | **33.7%** | **-47.6%** |

---

## 二、RL 训练选定的 8 个 Task（Qwen3-4B baseline）

| Task | Category | Grading | Qwen3-4B | qwen-plus | 提升空间 | 选择理由 |
|------|----------|---------|----------|-----------|---------|---------|
| task_02_stock | research | automated | 0% | 100% | 100% | qwen-turbo 也能 100%，纯行为策略问题 |
| task_12_skill_search | file_ops | automated | 0% | 100% | 100% | qwen-turbo 67%，可学 |
| task_10_workflow | complex | hybrid | 33% | 89% | 56% | 多步工具调用，RL 核心场景 |
| task_22_second_brain | memory | hybrid | 50% | 100% | 50% | 知识持久化，有部分能力可强化 |
| task_16_email_triage | organization | hybrid | 38% | 88% | 50% | 组织分类任务，提升空间大 |
| task_19_spreadsheet | data_analysis | hybrid | 20% | 60% | 40% | 数据分析，qwen-plus 也只 60% |
| task_18_market_research | research | hybrid | 44% | 93% | 49% | 研究类，需要 web_search 策略 |
| task_24_polymarket | research | hybrid | 21% | 67% | 46% | 研究+综合，需要多工具协作 |

**Category 分布**: research (3), complex (1), file_ops (1), organization (1), data_analysis (1), memory (1)

**Grading 分布**: hybrid (6), automated (2)

**平均提升空间**: 61.4%（qwen-plus 天花板 vs Qwen3-4B 当前分）
