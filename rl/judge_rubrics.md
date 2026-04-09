# Per-Task Judge Rubrics for Process Reward

通过对比 qwen-plus（成功）和 Qwen3-4B（失败/部分成功）的轨迹，提炼出 Qwen3-4B 的能力缺陷和每个 turn 的评判标准。

---

## 通用缺陷模式（Qwen3-4B 跨 task 共性问题）

| 缺陷 | 表现 | 规则检测方式 |
|------|------|-------------|
| **过早终止** | 只做 1-2 个 turn 就停了，该做的没做完 | `assistant_turns < expected_min_turns` |
| **不看结果就写** | web_search 返回后不分析，直接 write | write 之前没有纯文字分析 turn |
| **重复失败不换策略** | 同样的 sed 命令反复执行，每次都报错 | 相邻 turn 工具名+参数相同且上次 isError=true |
| **不验证产出** | write 之后不 read 回来确认 | write 后没有 read 同路径 |
| **内容太薄** | 写入的文件字数远少于 qwen-plus 的产出 | write content 字节数 < 阈值 |
| **用绝对路径** | 写 `/tmp/pinchbench/0018/...` 而非相对路径 | path 包含 `/tmp/pinchbench` |

---

## Task 02: Stock Price Research

**qwen-plus 策略**: web_search → web_fetch → 多次重试不同源 → 最终用合理数据 write
**Qwen3-4B 缺陷**: 只做了 1 个 assistant turn（web_search），搜到结果后直接停了，没有 write 文件

### Per-Turn Rubric

| Turn 行为 | reward | 检测规则 |
|-----------|--------|---------|
| 调用 web_search 搜索股价 | +0.15 | toolCall.name == "web_search" && args.query 包含 "stock" 或 "AAPL" |
| web_search 失败后换 query 重试 | +0.10 | 前一个 web_search 结果无有效价格，当前 turn 再次 web_search 且 query 不同 |
| 调用 web_fetch 获取详细数据 | +0.10 | toolCall.name == "web_fetch" |
| 调用 write 创建 stock_report.txt | +0.25 | toolCall.name == "write" && args.path 包含 "stock_report" |
| 写入内容包含价格+日期+摘要 | +0.15 | write content 匹配 `\$\d+` 且包含日期格式且 len > 200 |
| 只搜索不写文件就停了 | -0.30 | 最后一个 turn 无 write 且 task 未完成 |
| 编造股价（未经搜索确认） | -0.40 | write 之前没有任何 web_search/web_fetch |

---

## Task 12: Skill Search (Config File Update)

**qwen-plus 策略**: exec ls → read 逐个文件 → edit 精确替换（遇到歧义时扩大上下文）→ read 验证
**Qwen3-4B 缺陷**: 不读文件就用 `sed -i` 批量替换，macOS 上 sed 语法错误反复失败，3 次重试用同样的错误命令

### Per-Turn Rubric

| Turn 行为 | reward | 检测规则 |
|-----------|--------|---------|
| 先 exec ls 或 read 查看文件内容 | +0.15 | 第一个 tool 是 exec(ls) 或 read |
| 用 read 读取目标文件 | +0.15 | toolCall.name == "read" && path 包含 "config/" |
| 用 edit 精确替换 | +0.15 | toolCall.name == "edit" |
| edit 成功（toolResult 包含 "Successfully"） | +0.10 | toolResult.text 包含 "Successfully" |
| 替换后 read 验证 | +0.10 | edit 之后有 read 同一文件 |
| 不读文件直接用 sed 批量操作 | -0.30 | 第一个 tool 就是 exec(sed) 且之前没有 read |
| sed 失败后用相同命令重试 | -0.40 | 连续两个 exec 的 command 相同且前一个 exit code != 0 |
| 工具调用失败不换策略 | -0.30 | isError=true 后下一个 turn 用同样的工具和类似参数 |

---

## Task 10: Multi-step API Workflow

**qwen-plus 策略**: read config.json → 分析内容 → write Python 脚本 → write NOTES.md → 总结
**Qwen3-4B 缺陷**: 能完成但产出质量差，Python 脚本不够健壮，NOTES.md 内容单薄

### Per-Turn Rubric

| Turn 行为 | reward | 检测规则 |
|-----------|--------|---------|
| 先 read config.json | +0.15 | 第一个 toolCall.name == "read" && path 包含 "config.json" |
| 分析 JSON 内容（纯文字 turn 提取 endpoint） | +0.10 | assistant turn 包含 "endpoint" 或 "api.example.com" 的文字 |
| write Python 脚本 | +0.15 | toolCall.name == "write" && path 匹配 "*.py" |
| Python 脚本包含 requests + json + error handling | +0.10 | write content 包含 "import requests" 且 "import json" 且 "try" |
| write NOTES.md | +0.15 | toolCall.name == "write" && path 包含 "NOTES" |
| NOTES.md 内容充实（> 500 bytes） | +0.10 | write content 长度 > 500 |
| 不读 config.json 直接写脚本 | -0.20 | write *.py 之前没有 read config.json |

---

## Task 22: Second Brain (Knowledge Persistence)

**qwen-plus 策略**: read 现有笔记 → 理解结构 → write 新笔记条目 → 建立索引/链接
**Qwen3-4B 缺陷**: 能部分完成，但笔记组织不够好，缺少交叉引用

### Per-Turn Rubric

| Turn 行为 | reward | 检测规则 |
|-----------|--------|---------|
| 先 read 理解现有笔记结构 | +0.15 | 前几个 turn 有 read 操作 |
| 按任务要求 write 新笔记 | +0.15 | toolCall.name == "write" |
| 写入的内容包含结构化格式（标题/列表/链接） | +0.10 | write content 包含 "#" 或 "- " 或 "[[" |
| 多次 write 建立多个关联文件 | +0.10 | write 调用次数 >= 2 |
| 不读现有内容直接写 | -0.20 | 第一个 tool 就是 write 且之前没有 read |

---

## Task 16: Email Triage

**qwen-plus 策略**: read 所有邮件 → 分析优先级 → write 分类结果 → 逐封处理
**Qwen3-4B 缺陷**: 读取不完整，分类粗糙，处理步骤跳过

### Per-Turn Rubric

| Turn 行为 | reward | 检测规则 |
|-----------|--------|---------|
| read 遍历邮件目录/文件 | +0.10/次 | toolCall.name == "read" && path 包含 "email" 或 "mail" |
| 分析邮件内容（纯文字 turn 讨论优先级） | +0.10 | assistant turn 包含 "priority" 或 "urgent" 或 "分类" |
| write 分类/处理结果 | +0.15 | toolCall.name == "write" |
| 产出包含所有邮件的处理意见 | +0.15 | write content 覆盖多封邮件 |
| 只读了部分邮件就开始写结论 | -0.20 | read 次数 < 邮件总数的一半 |

---

## Task 19: Spreadsheet Data Analysis

**qwen-plus 策略**: read CSV → 尝试 read XLSX（二进制）→ 用 exec(awk) 计算 → write 报告
**Qwen3-4B 缺陷**: 读到 XLSX 二进制后没有有效处理，报告内容不基于实际数据

### Per-Turn Rubric

| Turn 行为 | reward | 检测规则 |
|-----------|--------|---------|
| read CSV 文件 | +0.15 | toolCall.name == "read" && path 匹配 "*.csv" |
| 用 exec 执行数据分析命令 | +0.15 | toolCall.name == "exec" && command 包含 "awk" 或 "python" 或 "cut" |
| exec 结果包含数值 | +0.10 | toolResult.text 匹配 `\d+` |
| 遇到 XLSX 二进制时换策略用 exec(python) | +0.15 | read XLSX 后的下一个 turn 用 exec + python |
| write 报告引用了实际计算结果 | +0.15 | write content 包含 exec 返回的数值 |
| 读到二进制乱码后不处理直接写报告 | -0.40 | read XLSX 返回乱码 → 下一步直接 write 不经 exec |
| 报告中的数字是编造的（不匹配 exec 结果） | -0.40 | write content 中的数值与 exec 返回的不一致 |

---

## Task 18: Market Research

**qwen-plus 策略**: 多次 web_search（市场概况→竞品对比→定价）→ write 长报告（8412 bytes）
**Qwen3-4B 缺陷**: 只搜索一次就写报告，搜索 query 用了过时年份（2023），报告太短（3269 bytes）

### Per-Turn Rubric

| Turn 行为 | reward | 检测规则 |
|-----------|--------|---------|
| 多角度搜索（>= 2 次 web_search） | +0.15 | web_search 调用次数 >= 2 |
| 搜索 query 包含当前年份 | +0.10 | args.query 包含 "2026" 或 "2025" |
| 搜索 query 包含具体公司/产品名 | +0.10 | args.query 包含具体公司名 |
| write 报告长度充实（> 5000 bytes） | +0.15 | write content 长度 > 5000 |
| 报告包含多个维度（市场规模、竞品、定价） | +0.15 | write content 包含 "市场" 和 "竞" 和 ("价" 或 "pricing") |
| 只搜索一次就写报告 | -0.20 | web_search 只调用 1 次 |
| 搜索 query 用过时年份 | -0.20 | args.query 包含 "2023" 或更早 |
| 报告太短（< 2000 bytes） | -0.20 | write content 长度 < 2000 |

---

## Task 24: Polymarket Briefing

**qwen-plus 策略**: 多次 web_search 搜不同话题 → 对每个 market 单独搜新闻 → write 综合报告
**Qwen3-4B 缺陷**: 搜索 query 用过时年份（2023），没有对每个 market 单独搜新闻，报告内容基于编造

### Per-Turn Rubric

| Turn 行为 | reward | 检测规则 |
|-----------|--------|---------|
| 搜索 Polymarket 趋势 | +0.15 | web_search query 包含 "Polymarket" 且 "trending" |
| 对每个 market 单独搜索新闻 | +0.10/次 | web_search query 包含具体 market 话题 |
| 搜索包含当前日期/近期时间 | +0.10 | args.query 包含 "2026" 或 "April" 或 "today" |
| write 报告包含 3 个 market | +0.15 | write content 包含 "## 1" 和 "## 2" 和 "## 3" |
| 每个 market 附带新闻来源 | +0.10 | write content 包含 URL 或 "Source" |
| 搜索用过时年份 | -0.20 | args.query 包含 "2023" 或更早 |
| 报告中编造市场数据（无搜索支撑） | -0.40 | write 前 web_search 返回无相关结果但报告写了具体数据 |

---

---

## Reference Trajectories（天眼：qwen-plus 成功路径）

从 qwen-plus 的成功轨迹中提取每个 task 的关键步骤序列。用于 Oracle 模式的 process reward 计算。

### task_02_stock

```yaml
expected_steps:
  - action: "web_search"
    pattern: "stock|AAPL|price|Apple"
    label: "搜索股价信息"
  - action: "web_fetch|web_search"
    pattern: "."
    label: "获取详细数据（可能需要多次重试不同源）"
    optional: true
    repeatable: true
  - action: "write"
    pattern: "stock_report"
    label: "写入报告文件"
  - action: null
    pattern: null
    label: "总结确认"
min_turns: 3
key_milestones:
  - "write stock_report.txt"
anti_patterns:
  - pattern: "write 之前没有任何 web_search"
    label: "编造内容"
    penalty: -0.20
qwen_plus_stats:
  turns: 7
  tool_calls: 6
  output_bytes: 745
```

### task_12_skill_search

```yaml
expected_steps:
  - action: "exec"
    pattern: "ls"
    label: "查看目录结构"
  - action: "read"
    pattern: "config/"
    label: "读取第一个配置文件"
  - action: "read"
    pattern: "config/"
    label: "读取第二个配置文件"
  - action: "edit"
    pattern: "config/"
    label: "精确替换第一个文件"
  - action: "edit"
    pattern: "config/"
    label: "精确替换第二个文件"
  - action: "read"
    pattern: "config/"
    label: "验证修改结果"
min_turns: 5
key_milestones:
  - "read config 文件"
  - "edit 或 write 修改配置"
  - "read 验证修改"
anti_patterns:
  - pattern: "exec sed -i"
    label: "用 sed 盲替换而不是 edit 精确替换"
    penalty: -0.15
  - pattern: "不读文件直接操作"
    label: "第一个 tool 就是 exec sed 且之前没有 read"
    penalty: -0.15
  - pattern: "重复相同的失败命令"
    label: "连续 exec 相同 command 且前一个失败"
    penalty: -0.20
qwen_plus_stats:
  turns: 9
  tool_calls: 8
  note: "遇到 edit 歧义时扩大上下文精确匹配，而非用 sed"
```

### task_10_workflow

```yaml
expected_steps:
  - action: "read"
    pattern: "config.json"
    label: "读取配置文件"
  - action: null
    pattern: "endpoint|api.example.com"
    label: "分析提取 endpoint（纯文字 turn）"
  - action: "write"
    pattern: "\\.py$"
    label: "写 Python 脚本"
  - action: "write"
    pattern: "NOTES|notes"
    label: "写文档"
min_turns: 3
key_milestones:
  - "read config.json"
  - "write *.py"
  - "write NOTES.md"
anti_patterns:
  - pattern: "不读 config.json 直接写脚本"
    label: "跳过信息收集"
    penalty: -0.10
content_quality:
  - file: "*.py"
    min_bytes: 500
    must_contain: ["import requests", "import json"]
  - file: "NOTES.md"
    min_bytes: 500
qwen_plus_stats:
  turns: 4
  tool_calls: 3
  py_bytes: 1704
  notes_bytes: 1389
```

### task_22_second_brain

```yaml
expected_steps:
  - action: "read|exec"
    pattern: "."
    label: "查看现有笔记结构"
  - action: "read"
    pattern: "."
    label: "读取现有笔记内容"
    repeatable: true
  - action: "write"
    pattern: "."
    label: "写入新笔记/更新索引"
    repeatable: true
min_turns: 4
key_milestones:
  - "read 理解现有结构"
  - "write 新笔记"
anti_patterns:
  - pattern: "不读现有笔记直接 write"
    label: "不了解上下文就操作"
    penalty: -0.10
qwen_plus_stats:
  turns: 7
  tool_calls: 4
```

### task_16_email_triage

```yaml
expected_steps:
  - action: "read|exec"
    pattern: "email|mail"
    label: "遍历邮件列表"
    repeatable: true
  - action: "read"
    pattern: "email|mail"
    label: "逐封读取邮件内容"
    repeatable: true
  - action: null
    pattern: "priority|urgent|分类"
    label: "分析优先级（纯文字 turn）"
  - action: "write"
    pattern: "."
    label: "写入分类处理结果"
min_turns: 5
key_milestones:
  - "read 所有邮件"
  - "write 分类结果"
anti_patterns:
  - pattern: "只读部分邮件就下结论"
    label: "信息收集不完整"
    penalty: -0.10
qwen_plus_stats:
  turns: 15
  tool_calls: 14
  note: "逐封邮件读取并分析，工作量最大的 task"
```

### task_19_spreadsheet

```yaml
expected_steps:
  - action: "read"
    pattern: "\\.csv"
    label: "读取 CSV 文件"
  - action: "read"
    pattern: "\\.xlsx"
    label: "尝试读取 XLSX（会得到二进制）"
  - action: "exec"
    pattern: "python|awk|pandas"
    label: "用命令行工具分析数据"
    repeatable: true
  - action: "write"
    pattern: "summary|report"
    label: "写入分析报告"
min_turns: 4
key_milestones:
  - "exec 计算得到实际数值"
  - "write 报告引用实际数据"
anti_patterns:
  - pattern: "读到 XLSX 二进制后直接 write 报告"
    label: "不处理数据就编造结论"
    penalty: -0.20
  - pattern: "报告数值与 exec 结果不一致"
    label: "编造数据"
    penalty: -0.20
content_quality:
  - file: "data_summary.md"
    min_bytes: 1000
    must_contain_numbers: true
qwen_plus_stats:
  turns: 7
  tool_calls: 6
  output_bytes: 1728
  note: "遇到 pandas 不可用时改用 awk，灵活切换"
```

### task_18_market_research

```yaml
expected_steps:
  - action: "web_search"
    pattern: "market|APM|observability"
    label: "搜索市场概况"
  - action: "web_search"
    pattern: "vs|comparison|Datadog|Splunk"
    label: "搜索竞品对比"
  - action: "web_search"
    pattern: "pricing|price"
    label: "搜索定价信息"
  - action: "write"
    pattern: "market_research"
    label: "写入研究报告"
min_turns: 4
key_milestones:
  - "web_search >= 2 次"
  - "write 报告 > 5000 bytes"
anti_patterns:
  - pattern: "web_search query 包含 2023 或更早年份"
    label: "搜索用过时年份"
    penalty: -0.10
  - pattern: "只搜索 1 次就写报告"
    label: "调研不充分"
    penalty: -0.10
content_quality:
  - file: "market_research.md"
    min_bytes: 5000
    must_contain: ["市场|market", "竞|competitor", "定价|pricing|price"]
qwen_plus_stats:
  turns: 5
  tool_calls: 4
  output_bytes: 8412
  note: "从三个角度搜索：市场领导者、竞品对比、定价模型"
```

### task_24_polymarket

```yaml
expected_steps:
  - action: "web_search"
    pattern: "Polymarket|polymarket"
    label: "搜索 Polymarket 趋势"
  - action: "web_search"
    pattern: "."
    label: "对每个 market 单独搜索新闻"
    repeatable: true
  - action: "write"
    pattern: "polymarket_briefing"
    label: "写入简报"
min_turns: 4
key_milestones:
  - "web_search >= 3 次（趋势 + 各 market 新闻）"
  - "write 报告包含 3 个 market"
anti_patterns:
  - pattern: "web_search query 包含 2023 或更早年份"
    label: "搜索用过时年份"
    penalty: -0.10
  - pattern: "write 内容包含编造的具体概率但搜索未返回该数据"
    label: "编造市场数据"
    penalty: -0.15
content_quality:
  - file: "polymarket_briefing.md"
    min_bytes: 1200
    must_contain: ["## 1", "## 2", "## 3"]
qwen_plus_stats:
  turns: 8
  tool_calls: 7
  output_bytes: 1719
  note: "先搜总体趋势，再对 3 个话题逐一搜新闻"
```

---

## Ablation 实验设计

三组对比实验，验证 process reward 和天眼的价值：

| 实验组 | 通用规则 | Reference Trajectory | Terminal Reward | 说明 |
|-------|:-------:|:-------------------:|:--------------:|------|
| **A: Baseline** | ✗ | ✗ | ✓ | 纯 terminal reward |
| **B: Rule-only** | ✓ | ✗ | ✓ | 通用行为规则，不看标准答案 |
| **C: Oracle** | ✓ | ✓ | ✓ | 拿着 qwen-plus 成功路径打分 |

### 预期结果

- **A vs B**: process reward 是否帮助模型在 terminal=0 的 task 上也能学习
- **B vs C**: 天眼是否加速收敛（尤其在 task_12 这种"方法选择"很关键的 task 上）
- 关注指标: PinchBench 总分、per-task 分数变化、收敛速度（多少 training step 后开始拿到 terminal reward）

### reward 计算方式

```
Mode A: reward[last_turn] = terminal_reward
        reward[other_turns] = 0

Mode B: reward[k] = generic_rule_reward(turn_k)
        reward[last_turn] += terminal_reward

Mode C: reward[k] = generic_rule_reward(turn_k) + oracle_reward(turn_k, reference)
        reward[last_turn] += terminal_reward
```

### reward 量级

- per-turn process reward 范围: [-0.50, +0.30]（放大以确保不被 terminal 淹没）
- terminal reward: {-1, +1}（任务失败 -1，成功 +1，天然有方向信号）
- process reward 是独立信号，不乘以 terminal reward
- 即使 terminal=-1，process reward 仍然生效，确保模型在困难 task 上也能学到"哪步做对了"
- 不需要 baseline：process reward 自带正负方向，terminal 自带 ±1 方向
