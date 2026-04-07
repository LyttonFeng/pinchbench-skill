# RL 训练 Task 变体

这里存放训练专用的 task 变体，**不是 PinchBench 测试集**。

## 原则

- PinchBench 原始 25 个 task = 测试集，**永远不放这里**
- 这里的 task 与测试集**同类型但不同 prompt**（换人名/时间/参数/措辞）
- grading 函数要跟着改，确保 reward 准确

## 命名规范

```
task_{原始ID}_{类型}_train_{序号}.md

示例：
  task_01_calendar_train_001.md   # calendar 类型变体 1
  task_01_calendar_train_002.md   # calendar 类型变体 2
  task_21_comprehension_train_001.md
```

## 格式

与 PinchBench task 格式完全相同，参考 `tasks/TASK_TEMPLATE.md`。

frontmatter 里加一个字段标记这是训练变体：

```yaml
---
id: task_01_calendar_train_001
name: Calendar Event Creation (Train Variant 1)
category: calendar
grading_type: automated
timeout_seconds: 120
workspace_files: []
train_variant: true     # 标记为训练变体
original_task: task_01_calendar
seed: 1                 # 变体 seed，决定 train/val/test split
---
```

## 当前变体列表

（待添加，benchmark 跑完后根据 hard bucket 任务生成）
