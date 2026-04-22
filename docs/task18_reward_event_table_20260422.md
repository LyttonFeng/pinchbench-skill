# Task 18 Reward Event Table

## 目标

把 `task_18_spreadsheet_summary` 的策略强行往这条轨道推：

- `quarterly_sales.csv` 可以直接 `read`
- `company_expenses.xlsx` 不要 `read`
- 对 `.xlsx` 走：
  - `exec`
  - `openpyxl` / `pandas.read_excel` / `pd.ExcelFile`
  - 抽出 workbook 结构
  - 计算 CSV + Excel 全部关键指标
  - 写 `data_summary.md`

不是只要“第一步必须对”，而是：

- 越早进入正确轨道越好
- 晚纠正也给 credit
- 但反复 `read .xlsx`、反复坏 `exec`、过早写假报告要强罚

---

## 满分轨迹

参考 run：

- [0114_qwen3-6-plus.json](/Users/lytton/work/reinforement_learning/pinchbench-skill/results/0114_qwen3-6-plus.json)
- [task_18_spreadsheet_summary.jsonl](/Users/lytton/work/reinforement_learning/pinchbench-skill/results/0114_transcripts/task_18_spreadsheet_summary.jsonl)

分数：

- `1.0 / 1.0`

### 满分轨迹的关键步骤

#### Turn 1

同时做两件事：

- `read quarterly_sales.csv`
- `exec` + `openpyxl.load_workbook('company_expenses.xlsx')`

而且 `exec` 直接把：

- `Q1_Expenses`
- `Budgets`
- 每行原始内容

全打印出来。

这是最关键的成功分叉点：

- 它没有去 `read company_expenses.xlsx`
- 而是第一轮就进入了正确的 workbook parsing path

#### Turn 2

第二个 `exec` 做精确聚合：

- CSV:
  - total revenue
  - total cost
  - total profit
  - total units
  - revenue by region
  - revenue by product
  - top region
  - top product
- Excel:
  - total expenses
  - expenses by dept
  - expenses by employee
  - top dept
  - top employee
  - budget vs actual

这一步不是“探索”，而是：

- 真正把所有 grader 关心的数都算出来

#### Turn 3

`write data_summary.md`

并且内容里有：

- 结构化表格
- CSV section
- Excel section
- insights section
- 正确数字

#### Turn 4

给用户总结，但这对 benchmark 不重要。

---

## 满分轨迹隐含的状态机

从这个满分轨迹看，task_18 可以拆成 4 个阶段：

1. `route`
   - 正确选择 `.xlsx -> exec parser`
2. `extract`
   - 拿到 workbook 结构和原始行
3. `compute`
   - 算出 CSV / Excel 全部关键指标
4. `report`
   - 写出正确 report

reward 应该围绕这 4 个阶段来设计。

---

## Reward Event Table

下面是建议版。

### A. Routing Events

#### A1. 首次对 `.xlsx` 直接 `read`

条件：

- tool=`read`
- 参数里包含 `.xlsx`

奖励：

- `-0.8`

说明：

- 这是当前 student 最致命的坏动作
- 必须强罚

#### A2. 重复对 `.xlsx` 直接 `read`

条件：

- 前面已经发生过一次 A1
- 再次 `read .xlsx`

奖励：

- `-0.4`

说明：

- 比第一次轻一点
- 但仍要惩罚死循环

#### A3. 首次正确进入 `exec + xlsx parser`

条件：

- tool=`exec`
- 命令里同时满足：
  - `.xlsx`
  - `read_excel` / `ExcelFile` / `openpyxl` / `load_workbook`

奖励：

- `+1.0`

说明：

- 这是最关键的强扭事件
- 必须给大正奖

#### A4. 在没有任何 `read .xlsx` 污染之前，直接进入 `exec + xlsx parser`

条件：

- 触发 A3
- 且之前没发生 A1

奖励：

- 额外 `+0.3`

说明：

- 奖励“一步走对”

#### A5. 先 `read .xlsx`，后面纠正成 `exec + parser`

条件：

- 前面已经发生 A1
- 后面首次触发 A3

奖励：

- 仍给 `+1.0`

说明：

- 这点很重要：
- 不能因为第一步错了，就不给后续纠正 credit

#### A6. 先 `read csv`，但还没进入 xlsx parser

条件：

- tool=`read`
- 参数里是 `.csv`
- 尚未触发 A3

奖励：

- `-0.1`

说明：

- 不是大错
- 但在这题里，它经常意味着 agent 在回避最难的 xlsx

---

### B. Extraction Events

#### B1. 成功拿到 workbook 结构

条件：

- `exec` 的 tool result 里出现至少两个结构性 marker：
  - `Q1_Expenses`
  - `Budgets`
  - `sheet`
  - `columns`
  - `Employee`
  - `Department`

奖励：

- `+0.5`

说明：

- 说明不是空跑 exec
- 而是真的读到了 workbook

#### B2. `exec` 尝试解析 xlsx 但失败

条件：

- tool=`exec`
- 命令看起来在解析 xlsx
- tool result 报错

奖励：

- `-0.25`

说明：

- 要罚
- 但不能罚太重，不然模型不敢尝试

#### B3. 同一个坏 `exec` 命令重复重试

条件：

- 当前 `exec` 参数与上一轮失败 `exec` 基本相同

奖励：

- `-0.25`

说明：

- 防止原地打转

---

### C. Compute Events

这些事件最好按“首次命中”给分。

#### C1. CSV 聚合结果被正确算出

子事件：

- total revenue
- total profit
- total units sold
- top region
- top product

每个首次命中：

- `+0.15`

说明：

- 5 个子项，总共 `+0.75`

#### C2. Excel 聚合结果被正确算出

子事件：

- total Q1 expenses
- top department
- top employee
- budget vs actual comparison

每个首次命中：

- `+0.2`

说明：

- 4 个子项，总共 `+0.8`

#### C3. 同时拿到完整 CSV + Excel 核心指标

条件：

- C1 五项都命中
- C2 四项都命中

奖励：

- 额外 `+0.4`

说明：

- 这是“算对全套指标”的 bonus

---

### D. Report Events

#### D1. 创建 `data_summary.md`

条件：

- `write data_summary.md`

奖励：

- `+0.1`

说明：

- 文件存在本身给一点点分
- 但不要给太多，不然容易 reward hack

#### D2. 在没有完成关键计算前就写 report

条件：

- 触发 D1
- 但 C1/C2 的大部分子项还没命中

奖励：

- `-0.3`

说明：

- 防止过早写假报告

#### D3. report 包含 CSV + Excel 两部分

条件：

- 报告文本里同时覆盖：
  - CSV/Sales
  - Excel/Expenses/Budget

奖励：

- `+0.2`

#### D4. report 数值接近正确

按 grader 关键字段逐项给：

- total revenue
- total profit
- top region
- top product
- total expenses
- top department
- top employee
- budget comparison

每项：

- `+0.15`

说明：

- 这里和 automated grading 基本对齐

#### D5. report 有 insights / synthesis

条件：

- 不只是罗列数字
- 还有跨 CSV + Excel 的总结

奖励：

- `+0.1`

---

## 强罚事件

这些建议 hard penalty。

### P1. 反复读取 `.xlsx` 二进制

- 每次 `-0.4`

### P2. 在没拿到结构前编造 Excel 数字

条件：

- 报告出现 Excel 指标
- 但此前没有 B1

奖励：

- `-0.6`

### P3. 写空壳报告

条件：

- 文件存在
- 但没有关键数字/没有 Excel section

奖励：

- `-0.4`

### P4. 长 thinking loop，无工具动作

条件：

- assistant 长文本
- 无 tool call

奖励：

- `-0.15`

---

## 最简实施版

如果先不做太复杂的解析，我建议第一版先落这 8 个事件：

1. `read .xlsx` -> `-0.8`
2. 首次 `exec + xlsx parser` -> `+1.0`
3. 无 `read .xlsx` 污染就直接 `exec` -> `+0.3`
4. 成功拿到 workbook structure -> `+0.5`
5. `exec` 解析失败 -> `-0.25`
6. 创建 `data_summary.md` -> `+0.1`
7. 过早写 report -> `-0.3`
8. terminal success -> 保留现有 terminal reward

这 8 个已经足够把：

- `.xlsx -> read`

往：

- `.xlsx -> exec parser`

强扭过去。

---

## 和当前 reward 的关键区别

当前 reward 更像：

- 每个 turn 的通用 progress score

这版 event table 更像：

- **task_18 的状态机奖励**

也就是：

1. 先把路走对
2. 再把结构拿到
3. 再把数算对
4. 再把报告写对

这个更适合“强扭一个关键行为边界”。

---

## 最后结论

如果目标是把 `task_18_spreadsheet_summary` 强扭到接近满分，
只奖励“有 tool call / 有进展”不够。

你必须显式 shaping 这几件事：

1. `.xlsx` 不要 `read`
2. 尽快进入 `exec + pandas/openpyxl`
3. 真正抽出 workbook 结构
4. 算出 CSV 和 Excel 的关键指标
5. 最后写对 `data_summary.md`

满分轨迹已经证明：

- 这题不是玄学
- 只要走对这条状态机，满分是能拿到的
