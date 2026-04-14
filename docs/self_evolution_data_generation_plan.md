# 自进化数据生产模块施工图

本文档记录 PinchBench / OpenClaw 在线 RL 的下一阶段施工方案：先用原始 `Qwen3-4B` 在真实 OpenClaw runtime 中生成失败轨迹，再用 teacher 分析失败原因并生成 hint / process reward rubric，最后导出 veRL 可训练数据。

## 1. 目标

第一阶段目标不是直接扩大训练，而是把数据生产闭环跑通：

- 用 `Qwen3-4B original` 作为主 agent，在 OpenClaw runtime 中真实执行任务。
- 收集多步 rollout trajectory、环境反馈、workspace 前后状态和 PinchBench grading 结果。
- 用 qwen-plus / SOTA teacher 做后台失败分析，不直接替代主 agent。
- 生成每条样本对应的 hint、failure type 和 process reward rubric。
- 导出当前 veRL + REINFORCE++ + LoRA 训练可直接消费的 parquet。

核心原则：

- `terminal reward` 复用 PinchBench outcome grading。
- `process reward` 来自 teacher 生成的 rubric，再由规则或 self-judge 执行。
- 训练和评测都走 OpenClaw runtime，保证 rollout / grading / update 的 runtime 一致。

## 2. 整体闭环

```text
task seed / task template
        |
        v
Qwen3-4B original agent rollout
        |
        v
OpenClaw runtime 多步执行
        |
        v
PinchBench grader 打分
        |
        v
失败轨迹分析
        |
        v
SOTA teacher 生成 hint / process rubric / failure diagnosis
        |
        v
构造 RL training sample
        |
        v
veRL + REINFORCE++ 更新 LoRA
        |
        v
新 LoRA 回到 OpenClaw runtime 评测
```

这里的 teacher 不是线上服务用户的主 agent，而是后台训练数据生产器。它负责诊断和生成训练信号，主 agent 仍然是待优化的 `Qwen3-4B` / LoRA policy。

## 3. 首批任务选择

先围绕两个适合 demo 和可视化的任务构造训练数据：

| 任务 | 选择原因 |
|---|---|
| `task_16_email_triage` | 多文件读取、分类、优先级判断、生成报告，适合展示多步执行轨迹 |
| `task_10_workflow` | 典型文件读取、配置解析、脚本生成、文档生成，能展示 baseline “只说不做” vs LoRA “实际执行” |

第一版建议每个 base task 生成 20 个训练变体，总计 40 条任务；每条任务跑 1-2 次 original rollout，总计约 40-80 条 trajectory。

## 4. 泛化数据构造

训练样本不能直接复制 RL8 测试题。需要构造同任务类型、不同实例的变体，让模型学到可泛化的执行策略。

### 4.1 `task_16_email_triage`

可变因素：

- 邮件数量：8 / 10 / 13 / 15 封。
- 邮件主题：客户请求、incident、newsletter、spam、内部审批、代码 review 等。
- 优先级分布：P0 / P1 / P2 / P4 比例变化。
- 文件命名：`email_01.txt`、`inbox_001.md`、按日期命名等。
- 输出要求：markdown 报告、CSV 汇总、按优先级分组、包含 daily plan。
- 干扰项：内容相似但优先级不同的邮件。

希望模型学到的能力：

- 先 inspect workspace。
- 读取所有相关邮件文件。
- 做分类、优先级和 recommended action。
- 写入指定报告文件。
- 最后 verify 输出文件存在且格式正确。

### 4.2 `task_10_workflow`

可变因素：

- 配置文件名：`config.json`、`settings.json`、`service.yaml`。
- endpoint 字段名变化：`api_endpoint`、`base_url`、`service_url`。
- 脚本名变化：`call_api.py`、`run_client.py`、`fetch_status.py`。
- API method：GET / POST。
- 文档文件名变化：`NOTES.md`、`README.md`、`RUNBOOK.md`。
- workspace 文件数量和目录结构变化。

希望模型学到的能力：

- 读取配置文件。
- 正确提取字段。
- 生成可运行脚本。
- 写文档。
- 检查文件内容是否满足要求。

## 5. 数据结构

建议保留完整训练样本信息，哪怕最终进入模型的字段较少。完整数据便于 debug、ablation 和复现实验。

```json
{
  "task_id": "task_16_email_triage_variant_001",
  "base_task": "task_16_email_triage",
  "prompt": "...",
  "workspace_seed": "...",
  "rollout_trace": [
    {
      "turn": 0,
      "model_response": "...",
      "tool_calls": [],
      "env_feedback": "OpenClaw exited before next request"
    }
  ],
  "grade": {
    "score": 0.0,
    "missing_requirements": [
      "did not read inbox files",
      "did not create triage_report.md",
      "hallucinated email contents"
    ]
  },
  "teacher_analysis": {
    "failure_type": "no_tool_call",
    "root_cause": "model described actions instead of executing tools"
  },
  "hint_rubric": {
    "before_action_hint": "Inspect files using tools before summarizing. Do not claim completion unless the required output file is written.",
    "process_reward_rules": [
      "positive if the agent lists relevant files",
      "positive if the agent reads all required input files",
      "positive if the agent writes the required output file",
      "negative if the agent claims completion without any tool call"
    ]
  },
  "reward_config": {
    "terminal_reward": "pinchbench_grader",
    "process_reward": "teacher_generated_rubric"
  }
}
```

## 6. 失败类型枚举

第一版先固定 failure taxonomy，避免 teacher 输出太自由。

```text
no_tool_call
missing_file_read
partial_file_read
wrong_file_write
hallucinated_completion
wrong_schema
wrong_priority_assignment
missing_verification
tool_error_not_recovered
```

这些 failure type 后续可以作为自进化系统里的能力边界信号：

- 稳定流程问题可以沉淀为 skill。
- 多步策略和行为偏好问题交给 RL。
- 工具缺失或 runtime 能力缺失需要扩展工具/skill，不适合只靠 RL。

## 7. 模块拆分

建议新增目录：

```text
rl/data_generation/
  schemas.py
  generate_task_variants.py
  run_original_rollouts.py
  analyze_failures.py
  build_hint_rubrics.py
  export_verl_dataset.py
```

### 7.1 `schemas.py`

定义 task variant、trajectory、grading result、teacher analysis、hint rubric、export sample 的数据结构。

### 7.2 `generate_task_variants.py`

从原始 RL8 task 派生训练变体。

输入：

```text
task_16_email_triage.yaml
task_10_workflow.yaml
```

输出：

```text
generated_tasks/task_16_email_triage_variant_001.yaml
generated_tasks/task_10_workflow_variant_001.yaml
```

约束：

- 不泄漏 RL8 测试答案。
- 保持任务类型一致。
- workspace 文件真实存在。
- expected output 可被 grader 自动判断。
- 难度覆盖 easy / medium / hard。

### 7.3 `run_original_rollouts.py`

启动 `Qwen3-4B original` vLLM，并让 OpenClaw 指向该 vLLM 作为主 agent。

每条任务输出：

```text
rollouts/<task_id>/
  trace.json
  openclaw_stdout.log
  workspace_before/
  workspace_after/
  grade.json
```

重点记录：

- model response。
- tool calls。
- env feedback。
- workspace before / after。
- PinchBench score。

### 7.4 `analyze_failures.py`

把失败轨迹交给 qwen-plus / SOTA teacher，生成结构化失败分析。

teacher 输入：

```text
task prompt
workspace before
model trajectory
workspace after
grader result
```

teacher 输出：

```text
failure_type
root_cause
missing_steps
recommended_process_hints
rubric_rules
```

### 7.5 `build_hint_rubrics.py`

把 teacher analysis 转成两类产物：

- 给训练 prompt 或 rollout context 使用的 hint。
- 给 reward manager 使用的 process reward rubric。

例子：

```text
Hint:
Before answering, inspect the workspace and use tools to read required files. Do not claim completion unless the required output file is written.

Process reward:
+0.1 if the agent lists relevant files
+0.2 if the agent reads required input files
+0.2 if the agent writes the required output file
-0.3 if the agent claims completion without any tool call
```

### 7.6 `export_verl_dataset.py`

导出当前 veRL 训练入口可直接读取的 parquet。

关键字段：

```text
prompt
data_source
ability
reward_model
extra_info
```

`extra_info` 至少包含：

```json
{
  "task_id": "...",
  "base_task": "task_16_email_triage",
  "task_yaml": "...",
  "workspace_files": "...",
  "hint_rubric": "...",
  "failure_type": "...",
  "reward_return_mode": "turn"
}
```

## 8. Reward 口径

Outcome reward 复用 PinchBench：

```text
terminal_reward = PinchBench grader score
```

Process reward 来自 teacher-generated rubric：

```text
process_reward = rule_checker_or_self_judge(trajectory, rubric)
```

第一版 turn-level reward 建议：

```text
中间 turn: process_reward_t
最后 turn: process_reward_t + terminal_reward
```

不再把整个 episode scalar 简单共享给所有 turn。这样可以更清楚地表达不同 turn 的行为质量。

## 9. 第一版验收标准

第一版数据生产模块完成后，需要满足：

- 能生成两个 base task 的训练变体。
- 能用 original `Qwen3-4B` 跑 OpenClaw rollout。
- 能保存 trajectory、workspace before/after、grading result。
- qwen-plus teacher 能输出结构化 failure analysis。
- 能生成 hint rubric。
- 能导出 veRL parquet。
- 1-step sanity 能读取新 parquet 并完成 `grading -> reward -> update -> ckpt`。

## 10. 施工顺序

建议按以下顺序实现：

1. 写 `rl/data_generation/schemas.py`。
2. 写 `generate_task_variants.py`，先支持 `task_16_email_triage` 和 `task_10_workflow`。
3. 写 `run_original_rollouts.py`，复用现有 OpenClaw agent loop / benchmark runner。
4. 先跑 5 条样本，确认 trace、workspace、grade 都完整。
5. 写 `analyze_failures.py`，接 qwen-plus 生成 failure analysis。
6. 写 `build_hint_rubrics.py`。
7. 写 `export_verl_dataset.py`。
8. 跑 1-step sanity。
9. 跑小规模 LoRA 训练。
10. 对两个 demo task 做 before / after 可视化。

## 11. 风险点

- 如果 original model 大量不发 tool call，训练数据会集中在 `no_tool_call`，需要先强化 tool-use hint 和 tool-call process reward。
- 如果 task variant 的 grader 不稳定，会污染 terminal reward。
- 如果 teacher rubric 太自由，reward 会不可复现，需要固定 failure taxonomy 和 rubric schema。
- 如果只围绕两个任务训练，demo 可以提升，但不能过度宣称泛化能力。
- 如果主 agent 是 Claude 等强 API 模型，则 LoRA 不能直接更新主模型；这时 RL 的对象应是可训练的 agent policy / 小模型，而不是 Claude 本体。

