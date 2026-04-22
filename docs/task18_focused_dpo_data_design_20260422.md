# task_18 focused DPO 数据设计（2026-04-22）

## 1. 这份文档解决什么问题

`task_18_spreadsheet_summary` 的 focused DPO，已经证明两件事：

1. 训推一致后，DPO 不是跑不起来。
2. 只做 `exec > read_xlsx` 一类 pair，不足以把首跳稳定钉成 `exec`。

原因很直接：

- 真实 OpenClaw runtime 下，模型首跳不是只会在 `exec` 和 `read_xlsx` 之间二选一。
- 它还会走别的“安全但错误”的动作，比如：
  - `session_status`
  - `read_csv_first`
  - `ls/find` 先探索

所以 focused DPO 的目标，不是只压掉一个错误动作，而是：

**在真实 runtime 的候选动作集合里，把 `exec + pandas.read_excel` 训练成最优首跳。**

---

## 2. 当前已经验证到什么程度

### 2.1 已经验证成功的部分

- 真实 runtime prompt 模板已经抓到过一版。
- 但后续 harness 实测说明，静态模板不是最终真相。
- 在真实三机链路里：
  - `Mac(ModelProxy) -> ECS(OpenClaw) -> L40S(vLLM)`
  - 当次 request 里实际抓到的是 `24 tools`
- 这说明 focused DPO 数据不能再只依赖旧的静态 `runtime_prompt_template.json`。
- 更可靠的做法是：
  - 以 harness 当次真实 request 抓回的 `messages + tools` 为准
- DPO 训练脚本已经修好：
  - completion 不再被 TRL 截断
  - `chosen/rejected logps` 非零
  - reward margin 会正常拉开

一轮 `1 epoch` focused DPO sanity 的训练日志已经证明：

- `logps/chosen` 从 `-255.2` 提到 `-159.1`
- `rewards/margins` 提到 `11.99`
- `train_loss` 降到 `0.06336`

这说明：

**focused DPO 训练链路本身已经通了。**

### 2.3 新的关键进展：harness-driven 数据采集已打通

这一步比前面所有离线模板都更关键。

已经确认：

- 真实三机链路可用：
  - Mac 负责 `ModelProxy`
  - ECS 跑 `OpenClaw`
  - L40S 跑 `vLLM + tool parser`
- `task_18` 的 harness-driven collector 已经能采到真实 first-step failure
- 已经落出一批真实 focused DPO 数据

当前真实 harness 数据统计：

- 共 `30` 条
- `read_csv_first: 29`
- `no_tool_call: 1`

说明：

- 当前 `Qwen3-1.7B-Instruct` 在真实 runtime 下，最稳定暴露出来的首跳错误，是：
  - 先读 `quarterly_sales.csv`
  - 再去读 `company_expenses.xlsx`
- 这批数据的真实性高于之前离线构造的 pair
- 但它的负例分布不够多样

### 2.2 还没解决的部分

虽然训练收敛了，但 direct sanity 里模型首跳还不是目标动作：

- 不是 `read_xlsx`
- 也还不是 `exec`
- 而是先走了 `session_status`

这说明当前 pair 设计还不够完整。

---

## 3. 为什么只做 `exec > read_xlsx` 不够

之前最自然的想法是：

- `chosen = exec + pandas.read_excel`
- `rejected = read company_expenses.xlsx`

这确实能压掉一类明显错误：

- 把 `.xlsx` 当文本文件直接读

但真实 runtime 里还有别的竞争动作没有被惩罚：

### 3.1 `session_status`

这是当前已经实测出现的首跳竞争动作。

特点：

- 看起来“安全”
- 不会立刻读坏上下文
- 很像 agent 的保守起手

如果训练里没有明确加入：

- `exec > session_status`

模型就可能学成：

- “我不要 `read_xlsx` 了”
- “但我先 `session_status` 也行”

这不是我们要的。

### 3.2 `read_csv_first`

这题 workspace 里同时有：

- `quarterly_sales.csv`
- `company_expenses.xlsx`

模型很容易先去读 CSV，因为：

- CSV 是纯文本，读取风险低
- base prior 里更像“正常分析的第一步”

但对 task_18 来说，这会把首跳从关键的 `.xlsx` 解析动作上带偏。

所以也要明确压：

- `exec > read_csv_first`

### 3.3 `ls/find/explore`

另一类常见保守动作是先探索目录：

- `ls`
- `find`
- `glob`

这类动作不一定完全错误，但在这道题里：

- 文件名已知
- 目标文件已知
- 第一跳探索是低效且偏题

所以也应该压：

- `exec > ls/find/explore`

---

## 4. focused DPO 的最小数据类型设计

### 4.1 结论先写在前面

对 `task_18`，最小建议不是 `1` 类，也不是 `3` 类，而是：

**至少 `4` 类 focused DPO 数据。**

### 4.2 最小可用 4 类

#### 类型 A: `exec > read_xlsx`

目标：

- 压掉最明显的坏动作
- 防止模型把 `.xlsx` 当文本文件读

chosen:

- 第一跳 `exec`
- 命令里包含：
  - `pandas`
  - `pd.read_excel`
  - 或 `pd.ExcelFile`

rejected:

- 第一跳 `read company_expenses.xlsx`

#### 类型 B: `exec > session_status`

目标：

- 压掉真实 runtime 下的保守替代动作

chosen:

- 第一跳 `exec`

rejected:

- 第一跳 `session_status(current/default)`

这是当前最重要的新增类型。

#### 类型 C: `exec > read_csv_first`

目标：

- 防止模型先处理 CSV，把关键首跳让掉

chosen:

- 第一跳 `exec` 解析 `.xlsx`

rejected:

- 第一跳 `read quarterly_sales.csv`

#### 类型 D: `exec > ls/find/explore`

目标：

- 压掉没必要的目录探索

chosen:

- 第一跳 `exec`

rejected:

- 第一跳 `exec ls`
- 或 `exec find`
- 或其他等价探索动作

---

## 5. 为什么是 4 类，不是 3 类

如果只做 3 类，最容易被省掉的是：

- `exec > read_csv_first`
或者
- `exec > ls/find`

但这两类都是真实存在的竞争动作。

从策略学习角度看，这不是“一个错误动作的修复”，而是：

**在多候选首跳里，建立一个明确的动作排序。**

至少当前这道题，排序应该接近：

1. `exec + pandas.read_excel`
2. 其他动作都更差

而“其他动作”至少已经确认包含：

- `read_xlsx`
- `session_status`
- `read_csv_first`
- `ls/find`

所以最小集合就是 4 类。

---

## 6. 可以扩展到第 5 类，但不是第一优先级

如果后续发现模型还会跑偏，可以再加：

### 类型 E: `exec > web_search / memory_*`

目标：

- 压掉与本题无关的外部检索或记忆调用

但这类优先级低于前 4 类，因为当前实测最强竞争动作不是它们。

所以建议顺序是：

1. 先做 4 类
2. 跑 direct sanity
3. 只有在首跳仍跑偏时，再补第 5 类

---

## 7. 每类需要多少数据

focused DPO 不需要很大。

---

## 8. 当前采用的训练集策略

现在不再等“纯真实 multitype 全收满”才训练。

原因：

- harness 数据已经足够真实
- 但目前主要集中在 `read_csv_first`
- 如果继续死等真实 `session_status / read_xlsx / explore_ls`，节奏太慢

所以当前采用一版 bootstrap 训练集：

### 8.1 训练集组成

文件：

- `rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_task18_harness_multitype_bootstrap_train_v2.jsonl`

分布：

- `read_csv_first: 29` 真实 harness
- `read_xlsx: 10` runtime-parity 补齐
- `session_status: 10` runtime-parity 补齐
- `explore_ls: 10` runtime-parity 补齐

总计：

- `59` 条

### 8.2 为什么这样混合是合理的

这 59 条不是随便拼的。

目的很明确：

1. 用真实 harness 数据锚定当前模型最真实的失败模式
2. 用已有 runtime-parity multitype 数据补齐其余竞争动作
3. 先尽快把第一版 multitype focused DPO 训起来

也就是说，这是一版：

**真实失败驱动 + 合成补齐的 bootstrap multitype focused DPO train set**

### 8.3 为什么去掉 `no_tool_call`

真实 harness 里有 `1` 条 `no_tool_call`。

这条先不进训练集，原因：

- 它的行为边界不够清晰
- 容易引入脏信号
- 当前更重要的是压明确的竞争动作

所以 bootstrap v2 明确去掉了它。

---

## 9. 当前阶段的结论

到这一步，可以下一个更硬的结论：

1. focused DPO 的训练框架已经通了
2. 训推一致必须走真实 harness
3. 真实 harness 数据已经能采
4. 当前真实失败主要是 `read_csv_first`
5. 第一版训练不再等满 100 条纯真实数据，而是先用 `59` 条 bootstrap multitype 集合开训

这一步的目标不是“一次到 90+”，而是：

- 先验证真实 harness 驱动的数据，是否比之前纯离线数据更能把首跳拉正

建议：

- 每类 `8 ~ 16` 对

所以第一版总量大概：

- `32 ~ 64` 对

设计重点不是堆量，而是：

- prompt 完全一致
- runtime 模板一致
- chosen / rejected 只改第一跳动作
- completion 简短、干净、无 tool result 污染

---

## 8. 数据构造原则

### 8.1 必须保持不变的部分

每个 pair 必须固定：

1. 同一份 runtime `system prompt`
2. 同一份 runtime `tools`
3. 同一份 `task_18` user prompt
4. 同一套 chat template 渲染方式

### 8.2 只允许变化的部分

只改：

- 第一条 assistant tool call

也就是：

- chosen 和 rejected 的唯一核心差异是“首跳选了什么动作”

### 8.3 不应该再混进去的污染

不要混入：

- continuation prompt
- 不同 session 状态
- 不同 tools schema
- 不同 system prompt
- 长 tool result
- `.xlsx` 二进制内容
- 多轮 assistant/tool 历史

focused DPO 的目的不是学完整任务，而是学首跳边界。

---

## 9. 推荐的实验顺序

### 第一步：4 类 focused DPO overfit sanity

目标：

- 证明在真实 runtime prompt 下
- 模型能把首跳稳定改成 `exec`

标准：

- direct sanity 连续多次首跳都是 `exec`
- 不再出现：
  - `read_xlsx`
  - `session_status`
  - `read_csv_first`

### 第二步：小规模泛化扩样

在 overfit sanity 过了之后，再做：

- 文件名变体
- prompt 表述变体
- workbook 结构变体

但 runtime system/tools 仍保持一致。

### 第三步：再决定是否接 RL

如果 focused DPO 已经能稳定解决：

- “首跳动作选择”

那后续 RL 更适合接管：

- 多步任务推进
- 中后段策略优化
- 长轨迹 reward 驱动改进

---

## 10. 当前最实在的结论

focused DPO 现在已经不是“要不要做”的问题，而是：

- **该怎么把负例集合定义完整**

对 `task_18`，当前最小可用设计是：

1. `exec > read_xlsx`
2. `exec > session_status`
3. `exec > read_csv_first`
4. `exec > ls/find/explore`

先把这 4 类打全，再看是否需要第 5 类：

- `exec > web_search / memory_*`

这比继续加 epoch、加 rank、或者盲目扩数据，更有价值。
