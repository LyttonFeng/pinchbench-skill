# 结论先说

## 1. 这个 task-specific shaping 怎么用

新的最小版 patch 在这里：

- [minimal_reward_patch.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/agent_loop/task18_event_reward/minimal_reward_patch.py)

它**不是替代整个 reward 系统**，而是：

- 在你当前每个 turn 已经算出来的 `base_turn_reward` 上
- 再加一层 `task_18` 专用 shaping delta

也就是：

```python
base_turn_reward
+ task18_patch_delta
= patched_turn_reward
```

然后再走你现有的：

- terminal reward
- per-turn token assignment
- EMA baseline normalization

所以最合理的接法是：

1. 现有 `rule / self-judge / oracle-judge` 先照常产一个 turn reward
2. 如果 `task_id == task_18_spreadsheet_summary`
   - 再调用 `apply_task18_minimal_patch_reward(...)`
3. 把两者相加
4. 再 clip 到更宽一点的范围
5. 后续流程不变

---

## 2. 你之前的 turn-level reward 还用不用

### 短答案

- **继续用**
- 但对 `task_18`，它不再是唯一信号

### 更准确地说

你当前的 turn-level reward 是：

- 通用 progress / judge score

这个 still useful，因为它还能提供：

- 有没有用工具
- 有没有推进任务
- 有没有重复 / 幻觉 / 空回答

但它的问题是：

- 对 `task_18` 最关键的行为分叉不够 sharp

所以我建议的结构是：

### 对 task_18：

```python
final_turn_reward =
    generic_turn_reward
  + task18_event_shaping
```

### 对其他 task：

```python
final_turn_reward = generic_turn_reward
```

这才是最稳的。

不要现在就把通用 turn-level 全删掉。

---

## 3. terminal reward 还要不要

### 要，而且必须继续用

因为 terminal reward 负责的是：

- 最终有没有成功
- 有没有真的把整题做完

task-specific shaping 解决的是：

- 路走对没
- 关键中间状态拿到没

而 terminal reward 解决的是：

- 最终结果对不对

所以这两者职责不同。

最合理的结构是：

### process / turn reward

- 负责局部决策
- 负责中间纠偏

### terminal reward

- 负责全局 outcome

也就是说：

- **task-specific shaping 不能替代 terminal reward**
- **terminal reward 也不能替代 task-specific shaping**

两者都要。

---

## 4. EMA mean 还有没有价值

### 结论

- **有价值**
- 但要看你现在训练是多任务还是单任务

你当前 EMA baseline 的作用是：

- 不同 task reward 水位不一样
- 用 EMA 做 task-level centering
- 防止某些 task 因为 raw reward 偏大，污染整体 advantage

### 如果你是多任务 RL8 训练

- **建议保留 EMA**

因为：

- task_16、task_18、task_24 的 reward 水位不会一样
- 你现在又要给 task_18 加更激进 shaping
- 不做 centering，task_18 很容易在 batch 里显得过重

### 如果你是只训 task_18 单任务

EMA 还有没有用？

- **有，但没那么关键**

因为：

- 不存在 task 间 reward scale 污染
- 但 EMA 仍然可以帮助：
  - 稳定 reward baseline
  - 减少“已经会了以后还持续大推”的问题

### 我对单任务 task_18 的建议

如果你要强扭 `exec`，我建议：

- 单任务阶段先保留 EMA
- 但把它调得更“慢一点”

比如：

- `PINCHBENCH_TASK_EMA_ALPHA` 更小
- 或者 baseline init 更低一点

原因：

- 你现在 reward patch 会让某些关键事件产生更大的 spike
- EMA 太快，会过早把这个 spike 吃掉

---

## 5. 这个 task-specific shaping 到底怎么接入

最推荐的接法是：

### Step 1

保留你现有：

- `generic_rule_reward(...)`
- 或 `self-judge` / `oracle-judge`

算出来的 `r`

### Step 2

如果是 task_18：

```python
patch_delta = apply_task18_minimal_patch_reward(
    task_id=task_id,
    turn=turn,
    prev_turns=prev_turns,
    tool_result=tool_result,
)
```

### Step 3

把它加上去：

```python
r = clip_task18_patched_turn_reward(r, patch_delta)
```

### Step 4

保留你现有：

```python
rewards = [r + terminal_reward for r in rewards]
```

### Step 5

保留你现有：

- `_normalize_turn_rewards(task_id, per_turn_rewards)`

---

## 6. 为什么我不建议“把原 turn-level 全换掉”

因为你现在要解决的是：

- `task_18` 的关键行为边界太弱

不是：

- 全系统 reward 框架完全错了

所以最经济的方式是：

- **在现有系统上叠加一个 task_18 hard patch**

而不是：

- 直接推倒重写所有 reward

工程上更稳。

---

## 7. 这个最小 patch 目前覆盖哪 8 个事件

在：

- [minimal_reward_patch.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/agent_loop/task18_event_reward/minimal_reward_patch.py)

当前最小 8 事件是：

1. 首次 `read .xlsx` -> `-0.8`
2. 重复 `read .xlsx` -> `-0.4`
3. 首次 `exec + xlsx parser` -> `+1.0`
4. 未污染前直接进入 parser -> `+0.3`
5. 先 `read csv` 但还没进 parser -> `-0.1`
6. parser result 显示 workbook structure -> `+0.5`
7. parser exec failed -> `-0.25`
8. 过早写 `data_summary.md` -> `-0.3`
   或 parser path 之后写 report -> `+0.1`

这 8 个事件的目的很纯：

- **先把 `.xlsx -> exec` 强扭过来**

不是一步到位解决所有数值错误。

---

## 8. 如果目标是“先把 exec 强扭过来”，最推荐配置是什么

### 推荐策略

#### 保留

- 当前 turn-level reward
- terminal reward
- EMA normalization

#### 新增

- task_18 minimal event patch

#### 顺序

```python
base_turn_reward
-> add task18 patch
-> add terminal reward
-> EMA normalize
-> token assignment
```

这是我认为最合理的第一版。

---

## 9. 如果只想单任务狠狠干 task_18，我会怎么配

### 我会这么干

1. 单任务训练 `task_18`
2. reward mode 仍然保留 `self-judge` 或 `rule`
3. 叠加 `task18 minimal patch`
4. terminal reward 保留
5. EMA 保留，但让它慢一点

为什么：

- self-judge / rule 给通用 progress
- patch 给关键行为边界
- terminal 给最终收口

三层一起上，最像“强扭 policy”。

---

## 10. 最后一句

你现在不该把旧 turn-level reward 全扔掉。

更合理的是：

- **旧 turn-level reward = 底座**
- **task_18 event shaping = 强扭层**
- **terminal reward = outcome 收口**
- **EMA = 防止 reward scale 发散**

也就是：

- 不是替换
- 是叠加

这个才是当前最稳的接法。
