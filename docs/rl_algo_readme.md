# Process-Shaped Turn RL

## A Lightweight RL Recipe for Live User Tool Agents

日期：2026-04-24

相关工程门禁：[`rl_data_runtime_checklist.md`](rl_data_runtime_checklist.md)

## TL;DR

这套算法的目标，不是追求“通用大规模 RL 配方”，而是解决一个更具体的问题：

> **当训练样本不多、reward 很 sparse、任务是 multi-turn tool agent、而且面向真实 live user case 时，怎样把 RL 做轻、做稳、做得可迭代？**

我们的答案是一个 **轻量 REINFORCE++ agent 变体**。核心取舍如下：

1. **不用 GRPO。**  
   这类任务的数据量不大，也不是“大批同 prompt 多采样、组内比较”的设定。每条样本更像一个真实用户 case，对应一次真实 agent episode。`rollout.n=1` 更自然。

2. **不用 critic，也不用 GAE。**  
   这些任务不是标准 MDP benchmark。turn 和 turn 之间没有干净、稳定、可学习的 value decomposition。很多关键收益来自：
   - 最终是否完成任务
   - 某个 turn 是否做对了关键 action
   - 是否写出最终 artifact  
   在这种场景里，强行上 critic / GAE 往往会把 credit assignment 搞脏。

3. **去掉 `masked_whiten`。**  
   训练样本少、reward 稀疏时，原生 `masked_whiten` 会把本来只属于少数关键 token 的 reward 扩散到整段 response token 上，造成 credit assignment 失真。对 agent 任务尤其不合适。

4. **不用 return-to-go 跨 turn 传播，改做 turn-level reward。**  
   当前算法更像“对每个 assistant turn 单独打分”，而不是假设所有 turn 之间存在严格、平滑、可 bootstrap 的价值传递。

5. **为了收敛，引入 task-specific EMA mean + var normalization。**  
   reward 在 task 间分布差异很大。我们按 task 维护 EMA mean 和 EMA variance，再做：

```text
(raw_reward - EMA_mean(task_id)) / sqrt(EMA_var(task_id) + 1.0)
```

这样既能减小 task 间 reward scale 漂移，又避免 sparse reward 下 advantage 爆炸。

6. **reward 采用 terminal + process reward 的组合。**  
   terminal reward 负责“任务最终成没成”，process reward 负责“中间 workflow 对不对”。  
   对 agent 任务，这两者缺一不可。

一句话总结：

> 这不是“把 PPO/GRPO 缩小”，而是**为小样本、多轮工具 agent、live user 场景专门裁出来的一套轻量 RL 方案**。

---

## 1. 问题设定

这里训练的不是单轮 QA，也不是纯文本 completion，而是 **真实 OpenClaw agent episode**：

1. 模型读当前 prompt
2. 生成 assistant 回复 / tool call
3. OpenClaw 执行工具
4. 工具结果进入下一轮上下文
5. 多轮后写出最终 artifact
6. 由 task grader 给出 terminal signal

所以训练对象不是“答案字符串”，而是：

- 一个多轮轨迹
- 一组 assistant turns
- 一份最终任务产物

这也是为什么很多标准 RLHF trick 在这里不合适：  
它们默认训练对象是单轮 response，而不是带 tool interaction 的 agent trajectory。

---

## 2. 为什么不用 GRPO / Critic / GAE

### 2.1 为什么不用 GRPO

GRPO 的强项是：

- 同 prompt 多采样
- 组内比较
- 不依赖显式 critic

但这里的训练样本更像：

- 每条样本都是一个 task case
- 每次 rollout 成本高
- episode 长
- 工具调用重

在这种条件下，`rollout.n > 1` 的性价比很差。  
更现实的设定是：

```text
每个 prompt -> 1 次真实 agent rollout
```

所以当前基座保留 **REINFORCE++**，但把它改造成 agent-task 版本。

### 2.2 为什么不用 critic

critic 适合在状态价值相对平滑、可近似、且未来回报有明确传播结构时工作。  
这里的任务不是这样：

- 某个 turn 可能只是读文件，没有即时价值
- 某个 turn 可能一条 write 就决定成败
- 某些 reward 完全取决于最终 artifact

这会导致 value head 很容易学成伪信号。

因此当前方案直接去掉 critic，不训练 value head。

### 2.3 为什么不用 GAE

GAE 默认一个前提：

> 后续回报可以通过 value function 和 bootstrapping 在时间上平滑传播。

但 agent task 的多轮工具交互里，这个前提并不稳。  
比如：

- 第 1 轮读邮件
- 第 2 轮又读邮件
- 第 3 轮突然写出最终报告

真实 credit 未必适合用标准 TD/GAE 去传播。

所以当前配置使用：

```text
gamma = 0.0
```

也就是不做 reward-to-go 传播，turn 奖励主要由 reward function 显式定义。

---

## 3. 算法核心：Turn-Level REINFORCE++

当前算法可以概括成：

```text
OpenClaw multi-turn rollout
-> turn-level reward
-> task-specific EMA normalization
-> token-span broadcast
-> REINFORCE++ update (no critic, no masked_whiten)
```

### 3.1 turn-level reward

每个 assistant turn 得到一个 scalar reward，而不是整条轨迹只在最后给一个总分。

这很关键，因为 agent 的很多行为是中间可判定的：

- 有没有重复 bulk read
- 有没有及时写 `triage_report.md`
- 有没有开始收敛
- 有没有写出结构化 artifact

如果只在 episode 末尾给一个 terminal scalar，这些过程信息会完全丢掉。

### 3.2 task-specific EMA mean + var normalization

对每个 task id 单独维护：

- `EMA_mean(task_id)`
- `EMA_var(task_id)`

然后对每个 turn reward 做：

```text
normalized_reward =
  (raw_reward - EMA_mean(task_id))
  / sqrt(EMA_var(task_id) + 1.0)
```

这一步的目的不是“做花哨归一化”，而是解决两个非常实际的问题：

1. **不同 task 的 reward scale 不同**  
   如果不做 task-specific normalization，一个 task 的大负分可能把另一个 task 的 baseline 拉歪。

2. **sparse reward 容易爆 advantage**  
   `sqrt(var + 1.0)` 中的 `+1.0` 是方差地板，用来避免 `sqrt(var + eps)` 在低方差阶段过度放大信号。

注意：

- 这里真正重要的是公式里的 `+1.0`
- 因此 `PINCHBENCH_TASK_EMA_VAR_INIT` 不再需要显式设置成额外超参

### 3.3 去掉 `masked_whiten`

原生 veRL `reinforce_plus_plus` 会做：

```python
advantages = masked_whiten(returns, response_mask)
```

对 sparse reward agent task，这一步问题很大：

- 原本只有少量关键 token 有 reward
- whiten 后，普通 token 也会拿到非零 advantage
- narration token / tool-call token /结构 token 的 credit 被混在一起

因此当前 patch 后的逻辑是：

```python
advantages = returns * response_mask
```

也就是：

- 只保留模型自己生成 token 的 reward
- 不再对整段 response 做二次 whitening

### 3.4 token-span broadcast

每个 assistant turn 先得到一个 scalar reward。  
然后再把它 broadcast 到该 turn 对应的 generated token span 上。

当前实现采用 **sum-preserving broadcast**：

```text
turn_reward / len(turn_generated_tokens)
```

这样做的好处是：

- 训练对象仍然是 token-level policy gradient
- 但 credit assignment 保留了 turn-level 语义

比旧的“只在 `<|im_end|>` 放一个点奖励”更合理。

---

## 4. Episode 数据流

单条 episode 的完整流程如下：

```text
train.parquet 采样一条 task prompt
    ->
OpenClawAgentLoop 启动 agent session
    ->
vLLM + LoRA policy 生成 assistant turn
    ->
OpenClaw 执行工具，tool result 回填上下文
    ->
重复多轮，直到任务结束
    ->
生成最终 artifact（如 triage_report.md）
    ->
task grader 计算 terminal reward
    ->
reward module 对每个 assistant turn 计算 process reward
    ->
task-specific EMA normalization
    ->
turn reward broadcast 到 token span
    ->
REINFORCE++ 更新 LoRA
```

这条数据流的重点是：

- **训练数据本身不是 trajectory**
- parquet 只提供 prompt / metadata
- trajectory 是在线 rollout 时实时生成的

---

## 5. Reward 设计：Terminal + Process Reward

这部分是整个算法的重点。

### 5.1 为什么需要 terminal + process reward

只用 terminal reward 的问题：

- 稀疏
- 训练太慢
- 看不出中间 workflow 哪一步错了

只用 process reward 的问题：

- 容易 reward hacking
- 模型可能学会“流程对了但最终任务没做成”

所以我们采用：

```text
总奖励 = process reward + terminal reward
```

其中：

- **terminal reward** 负责最终成败
- **process reward** 负责过程可控性

### 5.2 terminal reward

当前 task16 的 terminal raw 逻辑是：

```text
success:                    +1.0
failure but report exists:  -0.25
failure and no report:      -1.2
```

这背后的设计非常直接：

- 成功完成任务必须强正奖励
- 写了报告但质量差，说明 workflow 至少闭环，罚轻一点
- 连最终 report 都没写，说明发生了 `no-report collapse`，要重罚

### 5.3 process reward 的定位

process reward 不是在模拟最终 judge。  
它的作用是：

> **把“什么样的 agent workflow 更可能最终成功”编码出来。**

所以它更偏：

- workflow
- structure
- closure

而不是细粒度语义 judge。

---

## 6. Task16 Reward Shaping 逻辑

task16 当前使用：

- 文件：
  [reward_task16_event_only_v2.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/agent_loop/task16_event_reward/reward_task16_event_only_v2.py)

它的输入不是 LLM judge，而是：

- 当前 assistant turn
- 之前的 assistant turns
- 当前 turn 紧随其后的 tool result
- 最终 workspace artifact

### 6.1 它奖励什么

#### A. 报告闭环

如果当前 turn 写了 `triage_report.md`：

```text
+0.2
```

这是最基本的 closure 奖励。

#### B. 结构完整的报告

如果报告里显式出现：

- priority
- category
- action
- 且 P0-P4 覆盖足够

则给：

```text
+0.8
```

这一步是在逼模型产出一个**可 grading 的最终 artifact**，而不是只做思考或零散总结。

#### C. Incident-group schema

如果报告明确包含：

- `## Incident Groups`
- `## Standalone Items`

则再给：

```text
+0.2
```

这一步鼓励模型把 inbox-level 世界结构显式化。

#### D. 适时收敛

如果只做了少量 bulk read 就开始写报告：

```text
+0.5
```

这条规则的目的不是“越早写越好”，而是防止 agent 陷入：

- 无限 reread
- 一直搜集信息但不交付

#### E. 最终 artifact verifier

除了 turn-level event reward，v2 里还有一层 **metadata-driven final report verifier**：

- 检查是否覆盖 enough emails
- 检查 incident group schema
- 检查 `email_01 + email_13` 是否构成 P0 group
- 检查 `email_05` 是否接近 P1
- 检查 `email_08` 是否接近 P1

这一步不是完全 task-agnostic，但比简单关键词打分更稳定。

### 6.2 它惩罚什么

#### A. 重复 bulk read

如果历史上已经 bulk read 过一次，又继续大量读 inbox：

```text
-0.8
```

#### B. late no report

如果已经明显到了该写报告的时候，还继续 bulk read：

```text
-0.5
```

#### C. 连续只读不写

如果连续多个 assistant turn 都只在 read、不写报告：

```text
2 turns: -0.3
3 turns: -0.6
>=4 turns: -0.9
```

这是针对 `no-report collapse` 的核心约束。

#### D. post-coverage reread

如果已经覆盖够多不同 inbox 邮件，当前 turn 仍大量 reread：

```text
-0.35 / -0.7
```

这一步直接打的是“读过了还不收敛”。

#### E. generic / partial report

如果写了报告，但结构弱、P0-P4 不足、缺少关键信号：

```text
-0.6
```

#### F. context overflow

如果 tool result 明确显示：

- `maximum context length`
- 或输入超长

则给：

```text
-1.0
```

这是为了避免模型学会生成会把系统拖进 overflow 的轨迹。

### 6.3 这套 shaping 的本质

这套 shaping reward 的重点不是“完全还原人工 judge”，而是：

> **约束 agent workflow，鼓励它做出更有可能成功的轨迹。**

所以它更像：

- process-level shaping
- artifact-level verifier

而不是 oracle judge。

---

## 7. 为什么这套 reward 适合 live user case

对于 live user case，最重要的往往不是单轮语言漂亮，而是：

- agent 能不能闭环
- 会不会无限读工具结果
- 会不会忘写最终 artifact
- 会不会在关键结构上犯低级错误

所以 process reward 在这里很重要。  
它给模型的不是抽象偏好，而是很具体的 agent behavior bias：

- 读够就写
- 不要死循环 reread
- 最终一定交付 artifact
- artifact 要有结构

这类 bias 对小模型尤其重要。

---

## 8. 训练入口、脚本链路与关键文件

### 8.1 训练入口

主入口：

```bash
rl/train/run_reinforce_lora.sh
```

task16 wrapper：

```bash
rl/train/run_reinforce_task16_event_only_v2.sh
```

veRL 真正入口：

```bash
python3 -m rl.train.launch_main_ppo
```

### 8.2 脚本链路

```text
run_reinforce_task16_event_only_v2.sh
  -> 生成 task16 parquet
  -> 导出 task-specific env
  -> 调用 run_reinforce_lora.sh
  -> 组装 Hydra 覆盖项
  -> launch_main_ppo
```

### 8.3 关键文件

训练主链：

- [run_reinforce_task16_event_only_v2.sh](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/run_reinforce_task16_event_only_v2.sh)
- [run_reinforce_lora.sh](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/run_reinforce_lora.sh)
- [launch_main_ppo.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/launch_main_ppo.py)

reward / normalization：

- [reward_task16_event_only_v2.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/agent_loop/task16_event_reward/reward_task16_event_only_v2.py)
- [reward_manager.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/reward_manager.py)

agent rollout：

- [openclaw_agent_loop.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/agent_loop/openclaw_agent_loop.py)

veRL patch：

- [verl_no_masked_whiten_patch.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/verl_no_masked_whiten_patch.py)

数据生成：

- [build_task16_variant_prompts.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/build_task16_variant_prompts.py)

---

## 9. 当前推荐训练配置（Task16）

当前更稳的 task16 配置是：

```text
VERL_MODEL=Qwen/Qwen3-1.7B
TOTAL_TRAINING_STEPS=50
TOTAL_EPOCHS=4
TEST_FREQ=10
SAVE_FREQ=10
BATCH_SIZE=1
MICRO_BATCH=1
ROLLOUT_TEMPERATURE=0.8
VAL_DO_SAMPLE=True
VAL_TEMPERATURE=0.3
VAL_TOP_P=0.6
VAL_TOP_K=20
VAL_N=1
MAX_TURNS=8
MAX_PROMPT_LENGTH=16000
MAX_RESPONSE_LENGTH=4096
VLLM_GPU_MEM_UTIL=0.22
VLLM_MAX_NUM_SEQS=8
PINCHBENCH_REWARD_ASSIGNMENT=turn_broadcast
PINCHBENCH_NO_MASKED_WHITEN=1
PINCHBENCH_TASK_EMA_INIT=0.0
PINCHBENCH_TASK_EMA_ALPHA=0.05
```

其中最关键的不是“某个单独超参”，而是这个组合：

```text
no critic
no GAE
no masked_whiten
turn-level reward
task-specific EMA normalization
terminal + process reward
```

---

## 10. 这套算法的定位

这套方法的目标不是取代所有 RLHF / Agent RL 配方。  
它更适合以下条件：

- 数据量不大
- 每条样本是高价值 case
- rollout 成本高
- 任务是 multi-turn tool agent
- 需要快速做实验、快速回滚、快速迭代

所以它的定位更接近：

> **一种面向 live user case 的轻量、可解释、可迭代 agent RL 配方。**

这也是为什么本文的重点不是追求“更复杂的 RL 算法”，而是把 credit assignment、reward shaping 和 agent workflow 对齐。  
对这类任务来说，这比堆更重的 RL machinery 更重要。

---

## 11. Reproducibility Runbook

本节记录当前 task16 RL 实验的可复现路径：原始数据、parquet 生成、训练入口、关键环境变量、veRL patch 位置、以及 benchmark 测试方式。

建议把本文档作为主入口，不再另起碎片化说明文档。原因是：

- 前半部分解释算法为什么这么设计
- 本节解释这套算法具体怎么跑
- 两者放在一起，读者能从 design decision 直接追到工程实现

### 11.1 原始任务数据

task16 的 canonical 任务文件是：

- [tasks/task_16_email_triage.md](/Users/lytton/work/reinforement_learning/pinchbench-skill/tasks/task_16_email_triage.md)

训练数据不是离线轨迹，而是 **prompt pool**：

```text
prompt parquet row
  -> online OpenClaw rollout
  -> runtime trajectory
  -> terminal grader + process reward
  -> token-level reward tensor
```

也就是说：

- parquet 里存 prompt、task_id、extra_info
- 真实 trajectory 在线训练时由 OpenClaw + 当前 policy 生成
- reward 在 episode 结束后根据 transcript/workspace 计算

### 11.2 Task16 Prompt Pool 构造

数据生成脚本：

- [rl/train/build_task16_variant_prompts.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/build_task16_variant_prompts.py)

输出目录：

```text
rl/data/prompts_task16_variants/
  train.parquet
  val.parquet
```

当前生成规模：

```text
train: 71
val:   11
```

训练集组成：

```text
canonical: 1
base prompts: 30
targeted prompts: 40
```

targeted prompt groups：

```text
email13_coverage: 4
incident_linkage: 4
email13_priority: 4
bigclient_weighting: 4
security_weighting: 4
closure_and_stop_reread: 4
incident_graph: 4
priority_propagation: 4
report_schema_incident_groups: 4
report_schema_priority_fields: 4
```

每条 parquet row 的核心字段：

```text
data_source: task id / source name
prompt: OpenAI-style messages
ability: task id
reward_model: veRL compatibility field
extra_info:
  task_id
  prompt_group
  repeat_idx
  reward_rubric
```

其中 `extra_info.reward_rubric` 目前是 task16 共享 rubric，用于 reward / verifier 读取 task-specific expectations。它不是每条 prompt 独立人工 rubric；后续如果做 per-instance 数据，需要把这里升级成每个 synthetic case 自己的 rubric。

手动生成 parquet：

```bash
python3 rl/train/build_task16_variant_prompts.py \
  --tasks-dir tasks \
  --output-dir rl/data/prompts_task16_variants \
  --val-count 5
```

注意：`--val-count` 目前主要保留兼容；实际 val 选择会固定覆盖 canonical、少量 base prompt 和关键 targeted groups。

### 11.3 训练启动脚本

task16 推荐入口：

- [rl/train/run_reinforce_task16_event_only_v2.sh](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/run_reinforce_task16_event_only_v2.sh)

底层通用 LoRA RL 入口：

- [rl/train/run_reinforce_lora.sh](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/run_reinforce_lora.sh)

veRL 启动 shim：

- [rl/train/launch_main_ppo.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/launch_main_ppo.py)

启动链路：

```text
run_reinforce_task16_event_only_v2.sh
  -> build_task16_variant_prompts.py
  -> export task16 reward / data env
  -> run_reinforce_lora.sh
  -> patch veRL / Ray
  -> python3 -m rl.train.launch_main_ppo
  -> verl.trainer.main_ppo
```

### 11.4 当前推荐 Task16 训练配置

当前偏稳的 task16 训练配置：

```bash
export VERL_MODEL=Qwen/Qwen3-1.7B
export TOTAL_TRAINING_STEPS=50
export TOTAL_EPOCHS=4
export TEST_FREQ=10
export SAVE_FREQ=10

export BATCH_SIZE=1
export MICRO_BATCH=1
export LR=2e-5
export LORA_RANK=32
export LORA_ALPHA=64

export ROLLOUT_TEMPERATURE=0.8
export VAL_DO_SAMPLE=True
export VAL_TEMPERATURE=0.3
export VAL_TOP_P=0.6
export VAL_TOP_K=20
export VAL_N=1

export MAX_TURNS=8
export MAX_PROMPT_LENGTH=16000
export MAX_RESPONSE_LENGTH=4096
export VLLM_MAX_MODEL_LEN=32768
export VLLM_GPU_MEM_UTIL=0.22
export VLLM_MAX_NUM_SEQS=8

export REWARD_MODE=task16-event-only-v2
export PINCHBENCH_REWARD_RETURN_MODE=turn
export PINCHBENCH_REWARD_ASSIGNMENT=turn_broadcast
export PINCHBENCH_NO_MASKED_WHITEN=1
export PINCHBENCH_TASK16_TERMINAL_REWARD_WEIGHT=0.8
export PINCHBENCH_TASK_EMA_INIT=0.0
export PINCHBENCH_TASK_EMA_ALPHA=0.05
```

`PINCHBENCH_TASK_EMA_VAR_INIT` 可以不显式设置。当前 normalization 公式已经是：

```text
(raw_reward - EMA_mean) / sqrt(EMA_var + 1.0)
```

因此真正的方差地板来自公式里的 `+1.0`，不是来自 var init。

### 11.5 A100 / Pod 环境要求

训练 pod 需要：

```text
GPU: A100 80G 更稳；A40/L40S 可做推理测试
Python: 3.10+ / 3.12 已验证过部分环境
Core deps: verl, vllm, transformers, peft, ray, pandas, pyarrow
OpenClaw: 训练时需要远端 ECS OpenClaw 可 SSH 访问
Judge: DashScope qwen-plus API key，用于 PinchBench grading
```

关键环境变量：

```bash
export OPENCLAW_HOST=<ecs-ip>
export OPENCLAW_PORT=22
export OPENCLAW_USER=root
export OPENCLAW_SSH_KEY=/root/.ssh/id_ed25519

export DASHSCOPE_API_KEY=<dashscope-key>
export PINCHBENCH_GRADE_JUDGE_MODEL=qwen-plus
export PINCHBENCH_GRADE_JUDGE_BACKEND=api
export PINCHBENCH_GRADE_JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

持久化启动建议：

```bash
cd /workspace/pinchbench-skill
TRAIN_LOG_PATH=/workspace/pinchbench-skill/task16_event_only_v2_rl.log \
bash rl/train/run_reinforce_task16_event_only_v2.sh
```

日志默认路径：

```text
/workspace/pinchbench-skill/task16_event_only_v2_rl.log
```

checkpoint 默认目录：

```text
rl/checkpoints/reinforce_lora_task16_event_only_v2_qwen31/
```

每个保存步的 LoRA adapter 位于：

```text
global_step_XX/actor/lora_adapter/
  adapter_config.json
  adapter_model.safetensors
```

### 11.6 算法实现文件

Agent rollout / reward assignment：

- [rl/agent_loop/openclaw_agent_loop.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/agent_loop/openclaw_agent_loop.py)

关键逻辑：

```text
OpenClaw rollout
-> transcript/workspace
-> terminal grading
-> process reward
-> task EMA normalization
-> turn_broadcast reward tensor
```

Task16 process reward：

- [rl/agent_loop/task16_event_reward/reward_task16_event_only_v2.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/agent_loop/task16_event_reward/reward_task16_event_only_v2.py)

Reward manager：

- [rl/train/reward_manager.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/reward_manager.py)

veRL no-masked-whiten patch：

- [rl/verl_no_masked_whiten_patch.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/verl_no_masked_whiten_patch.py)

启动时会通过：

- [rl/train/launch_main_ppo.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/rl/train/launch_main_ppo.py)

自动 import patch。

当前对 veRL 的关键魔改：

```text
原生:
  advantages = masked_whiten(returns, response_mask)

当前:
  advantages = returns * response_mask
```

原因是 task-specific EMA normalization 已经在 reward 侧完成，再叠一层 masked whitening 会破坏 sparse / turn-level credit assignment。

### 11.7 Benchmark 测试脚本

RL8 benchmark：

- [scripts/run_bench_rl8.sh](/Users/lytton/work/reinforement_learning/pinchbench-skill/scripts/run_bench_rl8.sh)
- [scripts/run_bench_rl8_lora.sh](/Users/lytton/work/reinforement_learning/pinchbench-skill/scripts/run_bench_rl8_lora.sh)

benchmark 主入口：

- [scripts/benchmark.py](/Users/lytton/work/reinforement_learning/pinchbench-skill/scripts/benchmark.py)

task16 单题三遍测试推荐命令：

```bash
source ~/.pinchbench_env

MODEL=Qwen3-1.7B \
BASE_URL=http://127.0.0.1:18035/v1 \
PINCHBENCH_MODEL_TEMPERATURE=0.7 \
PINCHBENCH_MODEL_TOP_P=0.8 \
PINCHBENCH_MODEL_TOP_K=20 \
PINCHBENCH_OPENCLAW_MAX_TOKENS=4096 \
python3 scripts/benchmark.py \
  --model Qwen3-1.7B \
  --base-url http://127.0.0.1:18035/v1 \
  --api-key dummy \
  --suite task_16_email_triage \
  --runs 3 \
  --output-dir results \
  --no-fail-fast \
  --no-upload \
  --judge qwen-plus
```

如果是 LoRA checkpoint，需要先用 vLLM 起带 LoRA 的服务，再把 `MODEL` 改成 LoRA served name，例如：

```bash
export LORA_ADAPTER_PATH=/path/to/global_step_30/actor/lora_adapter
export VLLM_LORA_NAME=task16-rl-step30
export VLLM_PORT=8000
bash rl/scripts/start_vllm.sh Qwen/Qwen3-1.7B
```

本地 tunnel 后测试：

```bash
MODEL=task16-rl-step30 \
BASE_URL=http://127.0.0.1:18035/v1 \
PINCHBENCH_MODEL_TEMPERATURE=0.7 \
PINCHBENCH_MODEL_TOP_P=0.8 \
PINCHBENCH_MODEL_TOP_K=20 \
PINCHBENCH_OPENCLAW_MAX_TOKENS=4096 \
python3 scripts/benchmark.py \
  --model task16-rl-step30 \
  --base-url http://127.0.0.1:18035/v1 \
  --api-key dummy \
  --suite task_16_email_triage \
  --runs 3 \
  --output-dir results \
  --no-fail-fast \
  --no-upload \
  --judge qwen-plus
```

推荐固定的推理采样设置由 `scripts/lib_agent.py` 通过环境变量写入 OpenClaw model config；正式比较时需要在记录里写清：

```text
temperature=0.7
top_p=0.8
top_k=20
max_tokens=4096
judge=qwen-plus
runs=3
```

### 11.8 当前已知实验边界

这套 runbook 复现的是 **task16 targeted prompt pool + task-specific process reward** 的在线 RL。

它不能证明：

- reward 完全 task-agnostic
- prompt pool 每条样本都有独立 rubric
- 小模型已经学会通用跨邮件语义建模

它能证明的是：

- 在固定 task16 场景下，轻量 turn-level RL 可以稳定影响 agent workflow
- `no masked_whiten + turn_broadcast + task EMA` 是当前更合理的 credit assignment 链路
- process reward 对避免 no-report collapse、重复 reread、artifact 缺失有直接作用
