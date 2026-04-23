# task16 定向数据方案 v3

日期：2026-04-23  
任务：`task_16_email_triage`

## 目标

这版方案的核心结论是：

1. `reward` 不继续往细里写
2. 下一步主攻 **定向数据补充**
3. 重点不是泛化地“多造一些 prompt”，而是围绕真实失败模式，补能直接给 RL 用的 **prompt pool**

当前阶段的判断：

1. `task16 v2 reward` 已经把一部分过程控制问题打到了
2. 剩下的错误，主要是**关键邮件级别的语义判断不稳**
3. 这些错误更适合通过 **data support** 来补，而不是通过更细的 reward 去硬拉

---

## 一、当前最关键的失败集合

下一波数据必须直接围绕这些高频失败点展开。

### 1. `email_13` 覆盖不稳定

真实失败：

1. `latency alert` 被漏掉
2. 或者虽然提到，但非常弱

需要教会模型：

1. `email_13` 不能漏
2. 它不是可忽略的低优先级噪声

### 2. `email_01 + email_13` incident linkage 不稳定

真实失败：

1. outage 邮件和 correlated alert 被当成两个互不相关的事件

需要教会模型：

1. 这两封邮件属于同一个 operational incident cluster
2. 最终报告里要明确写出这种关联

### 3. `email_13` 优先级不稳定

真实失败：

1. `email_13` 被压成过低优先级

需要教会模型：

1. 活跃 outage 期间的 correlated monitoring alert，不是普通监控邮件
2. 它应该排在靠前的位置

### 4. `email_05` BigClient 权重不稳定

真实失败：

1. BigClient thread 被漏掉
2. 或提到了，但没有被提到足够高的优先级

需要教会模型：

1. 客户价值和 revenue risk 是重要信号
2. 高价值客户线程应该明显高于 routine internal mail

### 5. `email_08` security / compliance 权重不稳定

真实失败：

1. security/compliance item 被压低
2. 或者虽然识别出来，但没给明确 action

需要教会模型：

1. security/compliance deadline 需要更高优先级
2. 报告里应该有明确 next action

---

## 二、我们真正要做的不是 pair，而是 RL prompt pool

用户当前真正需要的是：

1. **RL 训练样本**
2. 即：online RL 可以直接吃的 `prompt pool`
3. 而不是只产 `chosen/rejected pairs`

这里要把几种数据形式分清楚：

### A. RL prompt pool

这是当前主线。

每一条 RL 样本本质上是：

1. 一个 prompt 变体
2. 同一个 canonical 任务
3. 同一个 workspace
4. 同一个 grading / rubric 体系

也就是说：

- **task 语义不变**
- **grader 不变**
- **只改 prompt emphasis**

这类数据可以直接产成：

- `train.parquet`
- `val.parquet`

并直接喂当前这条 online RL 训练脚本。

### B. chosen / rejected pairs

这类数据有价值，但不是当前 RL 主线的直接输入。

用途更适合：

1. DPO
2. ORPO
3. targeted SFT
4. rejection sampling / filtering

所以：

- `pair` 是辅助线
- `RL prompt pool` 才是当前主线

### C. 少量高质量 SFT 样本

也值得做，但优先级在 RL prompt pool 之后。

作用：

1. 提升 support
2. 稳定最终 `triage_report.md` 的结构和质量

---

## 三、RL prompt pool 应该怎么设计

现在的 `task16` RL 样本，不应该只是 generic prompt 改写。  
应该围绕 failure taxonomy，分组构造。

### 1. `email_13` coverage group

prompt 重点强调：

1. every inbox item must appear in the report
2. correlated alert cannot be omitted
3. low-salience但高价值的 operational signal 不能漏

### 2. `incident linkage` group

prompt 重点强调：

1. related outage / alert mails should be linked
2. related threads should not be treated independently
3. report should explicitly show grouping

### 3. `BigClient weighting` group

prompt 重点强调：

1. customer impact
2. revenue risk
3. high-value customer threads should be surfaced clearly

### 4. `security / compliance weighting` group

prompt 重点强调：

1. security/compliance deadlines need elevated priority
2. output must include concrete follow-up action

### 5. `closure / stop reread` group

prompt 重点强调：

1. one effective inbox pass
2. avoid reread loop
3. switch to `triage_report.md` once coverage is sufficient

### 6. `full coverage + structured artifact` group

prompt 重点强调：

1. every email accounted for
2. each item has:
   - priority
   - category
   - rationale
   - action
3. report is operator-ready

---

## 四、每条 RL 样本对应的 rule / rubric 怎么处理

这是关键点。

### 先说结论

**当前这批 RL prompt pool，不需要为每条 prompt 单独重写 grader。**

原因：

1. 它们仍然是同一个 canonical task：
   - `task_16_email_triage`
2. workspace 不变
3. terminal grading 仍然由 canonical task 的 automated check / rubric 来给
4. 变的是 prompt wording，不是任务定义

所以当前这条 RL 数据线的做法应该是：

### 对于每条 RL 样本

保留：

1. `task_id = task_16_email_triage`
2. canonical workspace
3. canonical grading_type
4. canonical rubric / checks

改变：

1. `prompt`
2. `prompt emphasis`
3. `extra_info` 中可记录这个样本属于哪类 failure group

也就是说：

- **rule / rubric 不是每条都重写**
- 而是每条样本都挂回 canonical `task_16_email_triage`

### 什么时候需要单独写 rubric

只有在你做的是：

1. 新任务实例
2. 新 inbox 内容
3. 新 workspace 文件
4. 新答案标准

那时才需要：

1. 新 markdown
2. 新 automated checks
3. 新 judge rubric

但当前这版不是。  
当前这版是：

- **同任务**
- **同 grader**
- **不同 prompt pool**

---

## 五、failure taxonomy + 样本生产的关系

这里要明确流程：

### 第一步：failure taxonomy

先把真实失败系统化。

建议字段：

1. `run_id`
2. `score`
3. `missing_emails`
4. `wrong_priorities`
5. `wrong_linkage`
6. `report_missing_or_partial`
7. `reread_loop`
8. `notes`

作用：

1. 统计哪些错误最频繁
2. 明确下一波 prompt pool 应该覆盖哪些 failure cluster

### 第二步：按 taxonomy 反推 RL prompt group

不是凭感觉写 prompt，而是：

1. 哪类错误高频
2. 就产哪类 prompt 变体

### 第三步：生成 RL 样本

输出为：

1. `train.parquet`
2. `val.parquet`

每行仍然是：

1. canonical task
2. variant prompt
3. 同一个 grading 体系

### 第四步：再考虑 pair / SFT 辅助线

如果后面需要更强 support，再补：

1. chosen/rejected pairs
2. 少量 targeted SFT

但它们是第二条线，不是当前 RL 主线。

---

## 六、建议的数据规模

### RL prompt pool

建议下一版至少做到：

1. canonical `1` 条
2. 每个 failure group `6-10` 条
3. 总量先做到 `30-50` 条

这比现在更有针对性，也仍然可控。

### pair / SFT 辅助数据

每个 target failure：

1. `5-10` 条 chosen/rejected
2. `3-5` 条强 SFT

但这部分可以放在 RL prompt pool 之后。

---

## 七、不要做什么

### 1. 不要只做 generic prompt 改写

如果只是换说法，不绑定真实失败点，价值有限。

### 2. 不要把 email-level semantics 全塞进 reward

这会带来：

1. 不稳定
2. 容易被 hack
3. prompt 一变就漂

### 3. 不要一上来为每条样本重写 grader

当前这条 RL 数据线不需要。

只要：

1. canonical task 不变
2. canonical workspace 不变
3. canonical rubric 不变

就可以直接复用 grading。

---

## 八、推荐执行顺序

### Phase 1

先建 `task16 failure taxonomy` 表。

### Phase 2

基于 taxonomy 产第一版 **targeted RL prompt pool**。

### Phase 3

生成新的：

1. `train.parquet`
2. `val.parquet`

### Phase 4

用这套新 prompt pool 重跑 RL，并和下列口径比较：

1. baseline
2. revised skill-like
3. current RL checkpoint
4. RL + targeted prompt pool

### Phase 5

如果还不够，再补：

1. chosen/rejected pairs
2. 少量 targeted SFT

---

## 九、预期结果

如果这条线有效，下一步的改进不该只是：

1. 更爱写 report

而应该具体表现为：

1. 更少漏掉 `email_13`
2. 更稳定地把 `email_01 + email_13` 连起来
3. 更稳定地提升 BigClient
4. 更稳定地提升 security/compliance

这才是 `task_16` 下一阶段真正该追的里程碑。*** End Patch
天天中彩票 to=functions.apply_patch code്commentary  大发官网commentary to=functions.apply_patch code  弘鼎***
