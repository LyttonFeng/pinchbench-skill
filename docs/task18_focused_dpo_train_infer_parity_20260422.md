# task_18 focused DPO 训推一致模板记录（2026-04-22）

## 1. 这份文档解决什么问题

`task_18_spreadsheet_summary` 之前的 SFT / DPO 实验，核心问题不是“模型学不会”，而是**训练时看到的 prompt 模板，和 OpenClaw 真实推理时发给 vLLM 的模板，不是一套东西**。

这份文档先把 3 件事固化下来，后续 focused DPO 都按这里做：

1. `task_18_spreadsheet_summary` 的原始 benchmark 题目。
2. 满分 chosen 轨迹应该长什么样，至少第一跳应该是什么。
3. 真实 runtime 的 prompt 组成：`system prompt + tools schema + user prompt + tool results`。

结论先写在前面：

- 之前用“简化 3 tools + 无 system prompt”做 SFT / DPO，**实验结论不可信**。
- exact-prompt overfit 在“训练模板一致”的小范围 direct sanity 下已经证明：**LoRA 不是完全没学到**。
- 下一步应该先做 **训推一致的 focused DPO**，而不是继续在简化模板上加数据、加 epoch、加 rank。

---

## 2. task_18 原始 benchmark 题目

来源：
- [task_19_spreadsheet_summary.md](/Users/lytton/work/reinforement_learning/pinchbench-skill/tasks/task_19_spreadsheet_summary.md)

原题摘要：

- workspace 里有两个文件：
  - `quarterly_sales.csv`
  - `company_expenses.xlsx`
- 需要读两个文件，计算：
  - CSV：`total revenue / total profit / total units / top region / top product`
  - Excel：`total Q1 expenses / top department / top employee / budget vs actual`
- 最终写到 `data_summary.md`

这个 task 的关键难点不在 CSV，而在：

- `.xlsx` 不是纯文本
- workbook 有多个 sheet
- agent 如果直接 `read company_expenses.xlsx`，通常会读出二进制垃圾
- 一旦上下文被二进制污染，后续很容易继续走错

所以对这个题，第一跳策略非常关键。

---

## 3. 满分 chosen 轨迹应该学什么

来源：
- [task18_dpo_experiment_20260420.md](/Users/lytton/work/reinforement_learning/pinchbench-skill/docs/task18_dpo_experiment_20260420.md)
- [collect_spreadsheet_runtime_rollouts.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/data_construction/collect_spreadsheet_runtime_rollouts.py)

已经确认的高质量 teacher 轨迹特征：

- teacher: `qwen3.6-plus`
- 在 synthetic spreadsheet variants 上做到过 `20/20` 满分
- 平均 `4.7 turns`

对 focused DPO 来说，真正该学习的不是整条长轨迹，而是下面这个边界：

### chosen 的关键行为

- 第一跳不要 `read company_expenses.xlsx`
- 第一跳应优先用 `exec`
- 在 `exec` 里用 Python 解析 workbook
- 典型命令应包含：
  - `pandas`
  - `pd.read_excel`
  - 或 `pd.ExcelFile`

### rejected 的关键行为

- 第一跳直接 `read company_expenses.xlsx`
- 或先 `read` 出 XLSX 二进制，再在脏上下文里继续乱写报告

这说明 focused DPO 的核心，不是“整条任务谁赢了”，而是：

**在同一个 runtime prompt 下，模型对第一跳应该偏向 `exec+pandas`，而不是 `read xlsx`。**

---

## 4. 这次真正定位到的根因

### 4.1 之前为什么一直觉得 SFT / DPO 没学会

我们之前观察到：

- exact-prompt overfit 的 train loss 可以压到很低
- 但一到 OpenClaw benchmark，首跳还是 `read`

一开始怀疑过：

- label masking 错了
- LoRA merge 坏了
- vLLM 没加载 adapter
- 数据量太少

这些都查过一轮，但都不是主因。

### 4.2 现在更可靠的结论

根因是**训推模板不一致**。

训练时我们一度使用的是：

- 简化版 `3 tools`
  - `read`
  - `exec`
  - `write`
- 无 OpenClaw runtime system prompt
- 工具 description / parameters 也都是简化版

但 OpenClaw 真实推理时给 vLLM 的是：

- 完整 runtime `system prompt`
- 完整 runtime tools schema
- 每个工具都带更长的 description / parameters
- 再加上当前会话上下文

这会导致：

- 模型在训练里学到的是：
  - “在 3 个简化工具下，看到 xlsx 用 exec”
- 但推理里模型看到的是：
  - 真实 runtime 工具集合
  - 不同的工具描述
  - 不同的 system prompt
  - 更长的 prompt 前缀

对 tool-calling 模型来说，这不是“小偏差”，而是**输入分布已经换题了**。

### 4.3 为什么这个解释更可信

因为它能同时解释三个现象：

1. 训练 loss 很低。
2. 在更接近训练模板的 direct sanity 下，LoRA 行为是会变化的。
3. 一回到 OpenClaw 真实 runtime，就退回 base model 的 `read` 先验。

所以当前判断是：

**之前的问题主要不是“模型没学会”，而是“训练没对着真实 runtime 学”。**

---

## 5. focused DPO 的可复用模板

后续 task_18 focused DPO，不再直接从“消息列表”出发，而是从“完整 runtime request 模板”出发。

### 5.1 模板的最小组成

每个 DPO pair 必须绑定这 4 部分：

1. `system_prompt`
2. `tools`
3. `user_prompt`
4. `first_assistant_action`

也就是：

```text
[OpenClaw system prompt]
[full tools schema]
[task_18 user prompt]
-> chosen/rejected differ only at first assistant action
```

### 5.2 chosen / rejected 应该怎么构造

#### chosen

- prompt 使用真实 OpenClaw runtime 的 system + tools + user prompt
- assistant 第一跳：
  - tool name = `exec`
  - arguments 里包含 workbook 解析逻辑
  - 优先使用 `pandas.read_excel` / `pd.ExcelFile`

#### rejected

- prompt 必须与 chosen 完全一致
- assistant 第一跳：
  - tool name = `read`
  - path 指向 `company_expenses.xlsx`

这样做，DPO 才真正只在学习：

- 同一个 runtime 条件下
- 第一跳选 `exec` 还是 `read`

而不是把别的变量也混进去。

### 5.3 什么东西不能再混进去

focused DPO pair 里，不应该再混入这些污染：

- `Continue where you left off`
- `session_status`
- 不同的 tools schema
- 不同的 system prompt
- 不同的 user prompt 重写
- chosen / rejected 不同的会话状态
- 不同长度的大段 tool results

尤其是 `.xlsx` 的 binary tool result，不应该放进 rejected completion 里主导长度和 token 分布。

---

## 6. 真实 runtime 模板应该怎么固化

这里是后续真正要复用的“源模板”定义。

### 6.1 source of truth

task_18 focused DPO 的 source of truth 不是手写 JSON，不是简化版 3 tools，而是：

- **OpenClaw 实际发给 vLLM 的 `/v1/chat/completions` request**

也就是要从 runtime 侧抓到：

1. 完整 `system prompt`
2. 完整 `tools` 数组
3. 真实 `messages`
4. `tool_choice=auto`
5. 其他与 tool-calling 有关的 request 字段

### 6.2 本仓库里已经有的接近位置

相关实现位置：
- [openclaw_agent_loop.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/agent_loop/openclaw_agent_loop.py)
- [model_proxy.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/agent_loop/model_proxy.py)

这两处说明了：

- OpenClaw 会向 proxy / vLLM 发真实 chat completions 请求
- 真正的训练一致模板，应该从这条链路里抽取

### 6.3 建议落库格式

建议把捕获结果单独固化成 JSON，而不是只写在代码注释里。

建议新增一类工件：

```text
rl/data/generated/task_18_spreadsheet_summary_runtime/runtime_prompt_template.json
```

建议字段：

```json
{
  "task_id": "task_18_spreadsheet_summary",
  "captured_at": "2026-04-22",
  "source": "openclaw_runtime_render",
  "system_prompt": "...",
  "tools": [...],
  "tool_choice": "auto",
  "notes": {
    "reasoning": "This template is the source of truth for parity training.",
    "tool_count": 17
  }
}
```

后续 SFT / DPO builder 都从这份模板出发，而不是自己手拼 3 tools。

### 6.4 2026-04-22 首次抓包结果

实际抓到的 runtime 模板路径：
- [runtime_prompt_template.json](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/data/generated/task_18_spreadsheet_summary_runtime/runtime_prompt_template.json)

本次抓包的关键信息：

- `tool_count = 17`
- `messages = [system, user]`
- `tool_choice = null`

当前这台 OpenClaw 的实际工具名是：

- `read`
- `edit`
- `write`
- `exec`
- `process`
- `cron`
- `sessions_list`
- `sessions_history`
- `sessions_send`
- `sessions_yield`
- `sessions_spawn`
- `subagents`
- `session_status`
- `web_search`
- `web_fetch`
- `memory_search`
- `memory_get`

这里有一个很重要的纠偏：

- 之前口头上一直说“18 tools”
- 但这次真正抓到的 source of truth 是 **17 tools**

后续训练、文档、benchmark 对齐，都应该以这份抓包 JSON 为准，不要继续口头假设 18。

---

## 7. 关于“完整 system prompt + 完整 18 tools schema”

这一点单独强调，因为它是这次回溯里最关键的坑。

### 7.1 之前犯的错

之前默认认为：

- 只要 user prompt 一样
- tools 名字大致一样
- 训练就算“差不多”

这个判断是错的。

对 function-calling 模型来说：

- `system prompt`
- `tools description`
- `tools parameters`
- `tool count`

这些都属于 prompt 本体。

所以：

- `3 tools` 和真实 runtime `17 tools`
- `无 system` 和 `有 OpenClaw system`

不是小差异，而是**不同任务分布**。

### 7.2 当前文档的状态

这份文档先把结论和模板规则固定下来。

完整 runtime `system_prompt` 和完整 tools schema，应以 **runtime 抓包 / render 输出** 为准，落到上面提到的：

- `runtime_prompt_template.json`

再由训练脚本直接消费。

换句话说：

- **这份 markdown 是“设计结论和约束”**
- **JSON 模板才是“训练输入的 source of truth”**

两者都需要保留。

---

## 8. focused DPO 对当前代码的直接影响

### 8.1 当前 `train_dpo_lora_fixed.py` 已经开始补 parity，但还没完整闭环

来源：
- [train_dpo_lora_fixed.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/train_dpo_lora_fixed.py)

当前 DPO 代码的状态：

- 已改成保留第一条 assistant 之前的完整 prompt 前缀，不再只留第一条 user
- 已改成用 `tokenizer.apply_chat_template(..., tools=tools)` 预渲染 prompt / chosen / rejected
- 仍然依赖 pair 数据本身携带正确的 runtime `tools`

这意味着：

- 如果 pair 数据本身没把 runtime tools/system 编进去
- 训练还是会退回“不完整 prompt 模板”

### 8.2 focused DPO 的实现要求

下一版 focused DPO builder / trainer 至少要满足：

1. pair 数据里显式携带 runtime `tools`
2. pair 数据里显式携带 runtime `system_prompt`
3. chosen / rejected 共享同一份 runtime prompt 前缀
4. 只在第一跳动作上做偏好学习

如果做不到这几点，task_18 的 focused DPO 还是会继续学偏。

---

## 9. 施工建议

推荐按这个顺序做：

1. 从 OpenClaw 真实请求里抓出 `system prompt + 完整 tools schema`
2. 固化成 `runtime_prompt_template.json`
3. 用这份模板重建 task_18 focused DPO pairs
4. chosen / rejected 只改第一跳：
   - chosen = `exec+pandas`
   - rejected = `read xlsx`
5. 再训 focused DPO
6. 先做 direct sanity
7. 再跑 `task_18_spreadsheet_summary` benchmark

---

## 10. 最后结论

这次 task_18 的关键发现可以压缩成一句话：

**不是模型没学会，而是之前训练的不是 OpenClaw 真实 runtime 这张卷子。**

所以从现在开始：

- focused SFT / focused DPO 的首要前提，不是多训一点，而是先把 `runtime prompt template` 固化成 source of truth。
