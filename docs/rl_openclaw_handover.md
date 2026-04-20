# JiuwenClaw-RL 工程交接文档

更新时间：2026-04-20

这份文档给后续接手 JiuwenClaw-RL 的工程同学看。当前原型链路是：

- `veRL`：训练框架。
- `Qwen3-1.7B` / Qwen 系列：策略模型。
- `OpenClaw runtime`：当前 agent runtime。
- `PinchBench`：任务集、grading、benchmark。

下一阶段目标：保留 `veRL + PinchBench`，把当前的 `OpenClaw runtime adapter` 替换成 `JiuwenClaw runtime adapter`。

## 一句话总结

我们实现了一个面向多轮工具 agent 的在线 RL 训练闭环：veRL 采样 PinchBench task，启动 agent runtime，收集工具调用轨迹，用 PinchBench grading 判断最终结果，把 terminal/process reward 对齐到 token-level reward tensor，然后用 REINFORCE++ 更新 LoRA policy。

## 当前整体架构

```text
veRL PPO/REINFORCE++ trainer
  -> custom AgentLoop
    -> 启动 runtime episode
       当前：OpenClaw on ECS
       后续：JiuwenClaw runtime
    -> runtime 发起 OpenAI-compatible chat/tool 请求
    -> AgentLoop 调 veRL vLLM rollout server
    -> 模型返回 text/tool calls
    -> runtime 执行工具并继续
    -> AgentLoop 收集 response tokens + tool trajectory
  -> PinchBench grading
  -> terminal reward + process reward
  -> custom veRL RewardManager 写 token-level rewards
  -> LoRA policy update
```

当前实现本质上是 OpenClaw 的 runtime adapter。真正可复用的是 veRL 侧契约：`AgentLoopOutput` 需要包含 prompt ids、response ids、response mask/logprobs、轨迹字段、token-level rewards。

## 重要文件

| 文件 | 作用 |
|---|---|
| `rl/train/run_reinforce_lora.sh` | 主训练入口。设置 veRL overrides、LoRA 配置、RewardManager、runtime 环境变量和预检。 |
| `rl/train/launch_main_ppo.py` | 启动 `verl.trainer.main_ppo` 前加载本地兼容 patch。 |
| `rl/agent_loop/openclaw_agent_loop.py` | 当前 OpenClaw runtime adapter。后续 JiuwenClaw 主要替换/参考这个文件。 |
| `rl/train/reward_manager.py` | 自定义 veRL RewardManager。把 per-turn/tool reward 转成 token-level reward tensor。这个要保留。 |
| `rl/agent_loop/reward.py` | terminal reward + process reward / PRM 逻辑和 task rubric。这个要保留并继续迭代。 |
| `sitecustomize.py` | 在 Ray worker 内自动加载 veRL/vLLM 兼容 patch。 |
| `scripts/benchmark.py`, `scripts/lib_agent.py`, `scripts/run_bench_rl8.sh` | 本地 Mac 跑 PinchBench RL8 benchmark 的路径。 |
| `docs/rl_openclaw_handover.md` | 本文档。 |

## 我们在 veRL 周边实现了什么

这个项目没有大规模 fork veRL。核心做法是：

- 继续使用 veRL 原生 PPO 入口。
- 注册一个自定义 multi-turn `AgentLoop`。
- 注册一个自定义 `RewardManager`。
- 通过 `sitecustomize.py` 给 Ray worker 加载少量兼容/debug patch。
- 通过 `rl/train/run_reinforce_lora.sh` 的 Hydra overrides 控制训练行为。

后续同学应该把下面这些理解成可复用的 veRL 改动。

### 1. 用 veRL PPO 入口跑 REINFORCE++

文件：`rl/train/run_reinforce_lora.sh`

我们保留 `verl.trainer.main_ppo` 作为 trainer，但把算法配置成 critic-free 的 REINFORCE++ 风格：

```bash
algorithm.adv_estimator=reinforce_plus_plus
algorithm.gamma=0.0
trainer.critic_warmup=0
algorithm.use_kl_in_reward=True
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

实际含义：

- 不训练 critic/value model。
- 每个 PinchBench episode 从外部环境拿 reward。
- 仍然加 KL，把 LoRA policy 约束在 base model 附近。
- 对 live-agent task 来说，这比 GRPO 更直接。GRPO 需要同一个 prompt 多次采样形成 group baseline，但 live runtime episode 慢、带状态、要执行工具，多采样会显著放大成本。

### 2. 通过 AgentLoop 接入多轮 runtime

文件：

- `rl/agent_loop/config.yaml`
- `rl/agent_loop/openclaw_agent_loop.py`

veRL 普通 rollout 是直接模型生成。我们走 veRL experimental multi-turn agent-loop 路径：

```bash
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.agent.default_agent_loop=openclaw_agent
actor_rollout_ref.rollout.agent.agent_loop_config_path=rl/agent_loop/config.yaml
```

当前 `OpenClawAgentLoop` 是 veRL 和外部 runtime 的桥：

- veRL 从 `train.parquet` 采样一条 task。
- AgentLoop 启动一个 runtime episode。
- runtime 请求模型补全。
- AgentLoop 通过 `server_manager.generate(...)` 调 veRL 的 vLLM rollout server。
- runtime 执行工具。
- AgentLoop 收集 generated token ids、mask、logprobs、transcript、grading result、rewards。
- AgentLoop 返回 `AgentLoopOutput` 给 veRL。

对 JiuwenClaw 来说，主要替换点就是这个 AgentLoop。算法不依赖 OpenClaw，算法依赖的是“runtime adapter 能返回同样格式的 `AgentLoopOutput`”。

### 3. Token-Level RewardManager

文件：`rl/train/reward_manager.py`

这是最核心的 veRL 侧 reward 改动。

为什么需要：

- veRL 默认 custom reward 更偏向每条样本一个 scalar reward。
- 工具 agent 轨迹里有很多 assistant turns。
- process reward / terminal reward 应该落到产生动作的 token 上，而不是只落在最后一个 token。

已实现行为：

- 定义 `PinchBenchRewardManager`。
- 从 `AgentLoopOutput.extra_fields["tool_rewards"]` 读取 token-aligned reward。
- 写入 veRL 的 reward tensor。
- 如果没有 `tool_rewards`，fallback 到 `turn_scores`。
- 如果 turn 对齐失败，再 fallback 到最后一个有效 token 的 scalar reward。

runtime adapter 需要尽量提供：

```python
extra_fields={
    "tool_rewards": [...],  # 长度与 response_ids 一致
    "turn_scores": [...],   # 每个 assistant turn 一个分数
    "task_id": "...",
    "trajectory": [...],
}
```

如果 JiuwenClaw 能提供 `tool_rewards`，训练就是预期的 token-level credit assignment。如果只提供 scalar score，训练仍能跑，但会退化成较弱的 final-token credit assignment。

### 4. Process Reward / PRM

文件：

- `rl/agent_loop/reward.py`
- `rl/agent_loop/openclaw_agent_loop.py`

runtime adapter 先用 PinchBench grading 得到 terminal success，再按 assistant turn 计算 process reward。

当前 reward 组成：

- `terminal_reward`：最终任务成功/失败信号。
- `process_reward`：每轮 assistant action 的质量分。
- `total_reward`：turn rewards 求和，最后一个 turn 上叠加 terminal reward。

当前模式：

```bash
REWARD_MODE=baseline
REWARD_MODE=rule
REWARD_MODE=self-judge
REWARD_MODE=oracle-judge
```

当前实验默认用 `self-judge`。如果要构造更干净的离线数据或更强标签，可以用 `oracle-judge`，通常接 `qwen-plus`。

### 5. Per-Task EMA Baseline

文件：

- `rl/train/reward_manager.py`
- `rl/agent_loop/openclaw_agent_loop.py`

RL8 不同 task 难度差别很大。全局 baseline 会让困难 task 长期显得差，让简单 task 长期显得好。

已实现：

- 按 `task_id` 维护 EMA baseline。
- reward 围绕该 task 最近均值做中心化。
- 默认环境变量：

```bash
PINCHBENCH_TASK_EMA_ALPHA=0.1
PINCHBENCH_TASK_EMA_INIT=0.1
```

注意：这不是 critic，只是一个轻量 per-task baseline，用来稳定稀疏 agent reward。

### 6. LoRA 训练和 checkpoint

文件：

- `rl/train/run_reinforce_lora.sh`
- `rl/verl_lora_only_ckpt_patch.py`
- `rl/verl_best_ckpt_patch.py`

当前训练是 LoRA-only：

```bash
actor_rollout_ref.model.lora_rank=32
actor_rollout_ref.model.lora_alpha=64
actor_rollout_ref.model.target_modules=...
```

围绕 veRL 增加的 checkpoint 行为：

- 保存可用于 serving/eval 的 LoRA adapter checkpoint。
- 可选按 validation reward 保留 best checkpoint。
- 可选保留 latest checkpoint，方便 debug。

这个很重要：PinchBench benchmark 评测时应该加载 LoRA adapter，而不是 FSDP 训练态的 full checkpoint。

### 7. veRL/vLLM 兼容 patch

通过 `sitecustomize.py` 自动加载：

- `rl/verl_debug_metrics_patch.py`
- `rl/verl_best_ckpt_patch.py`
- `rl/verl_lora_only_ckpt_patch.py`
- `rl/transformers_qwen3_5_patch.py`
- `rl/verl_qwen3_5_generation_patch.py`
- `rl/verl_vllm_lora_empty_guard_patch.py`

为什么用 `sitecustomize.py`：

- Ray worker 是独立 Python 进程。
- `PYTHONPATH` 指向 repo root 后，Python 会自动加载 `sitecustomize.py`。
- 这样 patch 能进入 `TaskRunner`、`WorkerDict`、vLLM server worker。

各 patch 的意图：

- `verl_debug_metrics_patch.py`：补 debug 可见性，避免 rollout/reward 指标为空时直接难查。
- `verl_best_ckpt_patch.py`：按 validation reward 保留 best checkpoint。
- `verl_lora_only_ckpt_patch.py`：保存轻量 LoRA adapter artifact。
- `transformers_qwen3_5_patch.py`：Qwen3.5 config/model 兼容 shim。
- `verl_qwen3_5_generation_patch.py`：处理 Qwen3.5 缺 `generation_config.json` 的情况。
- `verl_vllm_lora_empty_guard_patch.py`：如果 vLLM 收到空 LoRA，直接报清楚，不允许静默训练。

关键区分：

- 算法主体：AgentLoop、RewardManager、PRM、REINFORCE++ config。
- Qwen3.5 patch：兼容性 workaround。
- empty-LoRA guard：debug/safety patch，不是修复。如果触发，说明 LoRA tensor 没有同步到 vLLM，训练无效。

### 8. 当前已知 veRL 问题

Qwen3-1.7B：

- 当前比 Qwen3.5-2B 更适合作为训练主线。
- 当前栈里需要 `ROLLOUT_LAYERED_SUMMON=False`。
- 健康 LoRA sync 应该看到：

```text
collect_lora_params done count=392
```

Qwen3.5-2B：

- 遇到多处 veRL/vLLM/Transformers 兼容问题。
- 缺 `generation_config.json`，需要 patch。
- Qwen3.5 top-level config 包了 `text_config`，部分 vLLM 版本会误走 VL 路径。
- 不建议作为当前工程主线。

think-mode 长轨迹：

- Prompt FIFO 能防 prompt 超上下文。
- 但 response 太长时，backward 仍会 OOM。
- 最近 Qwen3-1.7B think 训练在 `MAX_TURNS=10` 下跑到 `global_step=6`，之后 backward OOM。
- 建议下一轮用 `MAX_TURNS=8`，必要时降低 `MAX_RESPONSE_LENGTH`。

## veRL 训练配置

训练脚本使用 veRL 0.7.1 风格 PPO 入口，但算法配置为 REINFORCE++：

```bash
algorithm.adv_estimator=reinforce_plus_plus
algorithm.gamma=0.0
algorithm.use_kl_in_reward=True
algorithm.norm_adv_by_std_in_grpo=False
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

LoRA 配置：

```bash
actor_rollout_ref.model.lora_rank=32
actor_rollout_ref.model.lora_alpha=64
actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_qkvz,in_proj_ba,in_proj_a,in_proj_b,in_proj_z,out_proj]
```

Rollout 配置：

```bash
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.agent.default_agent_loop=openclaw_agent
actor_rollout_ref.rollout.agent.agent_loop_config_path=rl/agent_loop/config.yaml
```

JiuwenClaw 迁移计划：

- 保留 veRL trainer 和 overrides。
- 新增一个 agent loop，例如 `jiuwenclaw_agent_loop.py`。
- 把 `default_agent_loop` 从 `openclaw_agent` 改成 JiuwenClaw loop。
- 保留 `PinchBenchRewardManager` 和 PinchBench grading contract。

## AgentLoop 契约

自定义 agent loop 必须返回 veRL `AgentLoopOutput`：

```python
AgentLoopOutput(
    prompt_ids=...,
    response_ids=...,
    response_mask=...,
    response_logprobs=...,
    reward_score=total_reward,
    num_turns=...,
    metrics=AgentLoopMetrics(...),
    extra_fields={
        "turn_scores": per_turn_rewards,
        "tool_rewards": reward_at_tokens,
        "total_reward": total_reward,
        "process_reward": process_reward,
        "terminal_reward": terminal_reward,
        "terminal_success": terminal_success,
        "task_id": task_id,
        "reward_mode": reward_mode,
        "trajectory": trajectory_for_reward,
    },
)
```

关键字段：

- `response_ids`：flatten 后的模型生成 token。
- `response_mask`：模型生成 token 为 `1`，环境/tool token 为 `0`。
- `response_logprobs`：模型 token logprobs；环境/tool token 可为 `0`。
- `tool_rewards`：token-aligned reward vector，长度与 response 一致。
- `turn_scores`：fallback 的 per-turn rewards。
- `trajectory`：assistant/tool transcript，用于 PRM 和 debug。

即使 JiuwenClaw runtime 接口和 OpenClaw 不同，也必须保持这个 veRL 侧契约。

## Runtime Adapter 职责

当前 `rl/agent_loop/openclaw_agent_loop.py` 做五件事：

1. 为一个 PinchBench task 启动 runtime episode。
2. 把 runtime 的模型请求路由到 veRL vLLM rollout server。
3. 把模型输出解析成 runtime 能理解的 text/tool calls。
4. 收集 transcript、response tokens、tool results、workspace state。
5. 跑 PinchBench grading，并附加 reward metadata。

JiuwenClaw 需要替换的 OpenClaw-specific 部分：

- 通过 SSH 启动 OpenClaw。
- OpenClaw agent config 生成。
- OpenClaw model provider 注册。
- OpenClaw transcript 解析。
- OpenClaw workspace sync。
- OpenClaw skill preflight。

需要保留的通用部分：

- Chat template application。
- vLLM `server_manager.generate(...)` 调用。
- Prompt FIFO / context compaction。
- Response token/mask/logprob 收集。
- Reward computation interface。
- `AgentLoopOutput` 格式。

## Prompt / Context 处理

多轮 agent prompt 很容易超过上下文。当前 adapter 已经有 FIFO compaction：

- `PINCHBENCH_AGENT_MAX_PROMPT_TOKENS` 默认等于 `MAX_PROMPT_LENGTH`。
- `_compact_messages_by_turn()` 会按完整 assistant turn 删除最旧历史，直到 prompt fits。
- 这个修过如下错误：

```text
Prompt length exceeds the model's maximum context length
```

重要区别：

- Prompt FIFO 解决 prompt overflow。
- 它不能解决生成 response 太长导致的 backward OOM。

JiuwenClaw 版本也应保留同样策略：

- 调 `server_manager.generate` 前限制 prompt tokens。
- 限制 max turns。
- 跟踪 response budget；如果 response tokens 超预算，要 compact 或 stop。

## Reward 系统

Reward 分两层：

1. PinchBench grading 给 terminal reward。
2. 每个 assistant turn 给 process reward。

### Terminal Reward

PinchBench grading 检查最终输出/workspace。当前转换逻辑：

- success：`+1.0 * PINCHBENCH_TERMINAL_REWARD_WEIGHT`
- failure：`0.0`
- 部分“声称完成但没有写文件”的场景：`-1.0`

文件创建失败等惩罚逻辑在 `rl/agent_loop/reward.py`。

### Process Reward

process reward 按 assistant turn 产生：

- `baseline`：不加 process reward。
- `rule`：规则打分。
- `self-judge`：当前本地模型自评。
- `oracle-judge`：更强外部 judge，通常是 `qwen-plus`。

当前默认：

```bash
REWARD_MODE=self-judge
```

PRM prompt 包含：

- task goal
- optional hints
- common mistakes
- previous actions
- current tool call
- current tool result preview

分数范围：

```text
-0.5 到 +0.2 / turn
```

terminal reward 会加到最后一个 turn。

## Token-Level Reward 对齐

最重要的 veRL 改动是 `rl/train/reward_manager.py` 里的 `PinchBenchRewardManager`。

存在原因：

- veRL 标准 custom reward 路径更像是把 scalar reward 放到最后一个 token。
- 多轮 agent RL 需要跨 turn 做 credit assignment。

它做的事：

- 创建一个和 `responses` 同 shape 的零 reward tensor。
- 优先读取 `extra_fields["tool_rewards"]`。
- 直接把 token-aligned reward 写进 tensor。
- fallback：把 `turn_scores` 放到 `<|im_end|>` token 位置。
- 再 fallback：把 scalar reward 放到最后一个有效 token。

因此 runtime adapter 应尽量提供 `tool_rewards`。

## Per-Task EMA Baseline

实现在 `rl/train/reward_manager.py`。

问题：

- RL8 task 难度不同。
- 全局 reward baseline 会让 hard task 干扰 easy task。

方案：

- 每个 `task_id` 维护 EMA baseline。
- reward 归一化为：

```text
advantage-like score = raw_reward - EMA(task_id)
```

关键环境变量：

```bash
PINCHBENCH_TASK_EMA_ALPHA=0.1
PINCHBENCH_TASK_EMA_INIT=0.1
```

这对多任务 agent 训练很重要，迁移到 JiuwenClaw 时应保留。

## PinchBench 集成

PinchBench 用在两处：

1. 训练 prompt 数据：
   - `rl/data/prompts/train.parquet`
   - `rl/data/prompts/val.parquet`

2. Grading：
   - episode 结束后，用 PinchBench grading 评估最终 workspace/transcript。
   - benchmark 从本地 Mac 通过 `scripts/run_bench_rl8.sh` 跑。

训练脚本会先做 train-vs-benchmark prompt parity check：

```bash
python3 rl/scripts/check_train_infer_parity.py
```

这个检查要保留。它防止训练 prompt 和 benchmark prompt 不一致。

## Benchmark 流程

Benchmark 不在 RunPod 里跑，在本地 Mac 跑。RunPod 只负责模型推理。

典型流程：

```bash
# 1. 在 RunPod serve model 或 LoRA
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --port 8021 \
  --max-model-len 40960 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# 2. Mac 到 RunPod 建 tunnel
ssh -N \
  -L 127.0.0.1:18021:127.0.0.1:8021 \
  root@<pod-ip> -p <pod-port> -i ~/.ssh/id_ed25519

# 3. Mac 跑 RL8 benchmark
source ~/.pinchbench_env
MODEL=Qwen3-1.7B BASE_URL=http://127.0.0.1:18021/v1 bash scripts/run_bench_rl8.sh
```

tool parser 很关键：

- Qwen3 / Qwen3-1.7B：通常用 `hermes`。
- Qwen3.5：必须用 `qwen3_xml`。
- Qwen3.5-2B：不要把 `--reasoning-parser deepseek_r1` 和 `qwen3_xml` 混用；它可能把 tool call 吞进 reasoning，导致 `tool_calls=[]`。
- 一定要开 `--enable-auto-tool-choice`。

## 模型 / 实验状态

### Qwen3-1.7B Benchmark

已观察到的 RL8 benchmark：

| 模型设置 | RL8 分数 |
|---|---:|
| Qwen3-1.7B non-think | 34.5% |
| Qwen3-1.7B think | 53.0% |

详情见：

- `docs/qwen3_1_7b_rl8_think_vs_nonthink_20260419.md`

### Qwen3-1.7B 训练

模型可以初始化、rollout、同步 LoRA，并训练若干 step。

必要设置：

```bash
ROLLOUT_LAYERED_SUMMON=False
```

原因：

- `layered_summon=True` 时，当前 FSDP/PEFT 命名下 veRL 对 Qwen3-1.7B 收集到 0 个 LoRA tensor。
- 正确同步应看到：

```text
collect_lora_params done count=392
```

当前问题：

- think-mode 训练在长轨迹下会在 `loss.backward()` OOM。
- `MAX_TURNS=16` OOM。
- `BATCH_SIZE=1 + MAX_TURNS=10` 跑到 `global_step=6` 后 OOM。
- 最近失败日志：

```text
/tmp/train_qwen31_think_16_bt1_turn10.log
```

OOM signature：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ~8.04 GiB
```

原因：

- think mode 会生成长 reasoning。
- 长工具轨迹会产生 20k+ training sequence。
- backward peak memory 超过 44GB GPU。

建议下一轮训练：

```bash
BATCH_SIZE=1
MICRO_BATCH=1
MAX_TURNS=8
MAX_RESPONSE_LENGTH=<必要时低于 12000>
ROLLOUT_LAYERED_SUMMON=False
OPENCLAW_MODEL_REASONING=1
```

或者先跑 no-think 稳定 LoRA baseline：

```bash
OPENCLAW_MODEL_REASONING=0
```

### Qwen3.5-2B

当前不建议继续作为主线。

遇到的问题：

- Transformers/veRL/vLLM 对 `qwen3_5` 的兼容问题。
- 缺 `generation_config.json`。
- Qwen3.5 top-level config 包了 `text_config`。
- vLLM LoRA sync 不稳定。
- 初始化时 host RAM 压力大。

已有兼容 patch：

- `rl/transformers_qwen3_5_patch.py`
- `rl/verl_qwen3_5_generation_patch.py`
- 由 `sitecustomize.py` 加载

这些是 debug 辅助，不是完全稳定的生产路径。

## veRL / vLLM Patches

这些 patch 通过 `sitecustomize.py` 自动加载，因为 Ray workers 会继承 `PYTHONPATH`。

Patch 列表：

- `rl/verl_debug_metrics_patch.py`
  - 避免空 rollout probability diff metrics 导致难查问题。
- `rl/verl_best_ckpt_patch.py`
  - 按 validation metric 保留 best checkpoint，并可选保留 latest。
- `rl/verl_lora_only_ckpt_patch.py`
  - 保存更小的 LoRA adapter checkpoint。
- `rl/transformers_qwen3_5_patch.py`
  - Qwen3.5 config/model compatibility shim。
- `rl/verl_qwen3_5_generation_patch.py`
  - Qwen3.5 generation config fallback。
- `rl/verl_vllm_lora_empty_guard_patch.py`
  - 增加诊断；如果 LoRA collection 为空，明确 hard fail。

重要：`verl_vllm_lora_empty_guard_patch.py` 现在会在 LoRA 为空时直接失败，这是有意的。静默继续会让训练变成无效训练。

## 已知 veRL site-package 直接 patch

`masked_whiten` epsilon patch 目前还是直接改了安装环境里的 veRL：

```python
# /usr/local/lib/python3.12/dist-packages/verl/utils/torch_functional.py
whitened = (values - mean) * torch.rsqrt(var + 1.0)
```

如果没有这个改动，稀疏 token reward 容易产生很大的 advantage。

后续应把它沉淀成 repo patch 或 upstream config option。

## 当前训练命令模板

Qwen3-1.7B think 的相对安全版本：

```bash
cd /workspace/pinchbench-skill
OPENCLAW_HOST=8.163.82.224 \
OPENCLAW_PORT=22 \
OPENCLAW_USER=root \
OPENCLAW_MODEL_REASONING=1 \
VERL_MODEL=Qwen/Qwen3-1.7B \
RUN_VERSION=qwen31_think_bt1_turn8 \
TOTAL_TRAINING_STEPS=16 \
BATCH_SIZE=1 \
MICRO_BATCH=1 \
MAX_TURNS=8 \
REWARD_MODE=self-judge \
PINCHBENCH_TERMINAL_REWARD_WEIGHT=1.0 \
PRM_MODEL=Qwen/Qwen3-1.7B \
PRM_VLLM_BASE_URL=http://localhost:8000/v1 \
HYDRA_FULL_ERROR=1 \
ROLLOUT_LAYERED_SUMMON=False \
VLLM_MAX_MODEL_LEN=40960 \
bash rl/train/run_reinforce_lora.sh 2>&1 | tee /tmp/train_qwen31_think_bt1_turn8.log
```

## JiuwenClaw 需要实现什么

JiuwenClaw runtime adapter 要实现和当前 OpenClaw adapter 一样的逻辑接口：

1. 为一个 task 启动 runtime episode。
2. 把 task prompt 喂给 JiuwenClaw。
3. 接收 JiuwenClaw 的模型请求。
4. 把这些请求转换成 veRL/vLLM prompt ids。
5. 把模型输出作为 text/tool calls 返回给 JiuwenClaw。
6. 收集 tool trajectory 和 workspace state。
7. 执行 PinchBench grading。
8. 返回包含 token ids、masks、rewards、extra fields 的 `AgentLoopOutput`。

不要一开始重写 veRL 训练。最快路径是只替换 runtime adapter，同时保留：

- `run_reinforce_lora.sh`
- `PinchBenchRewardManager`
- `reward.py`
- PinchBench grading
- benchmark scripts
- LoRA checkpoint patches

## 训练时重点看什么

健康日志：

```text
collect_lora_params done count=392
Training Progress: ...
training/global_step:N
actor/grad_norm:<finite small number>
critic/advantages/max:<finite>
Chat template done, prompt_ids=<below budget>
```

异常日志：

```text
collect_lora_params done count=0
empty LoRA model reached vLLM merge path
Prompt length exceeds the model's maximum context length
torch.OutOfMemoryError during loss.backward
assistant content: []
tool_calls=[]
```

## 推荐下一步

1. 复制 `OpenClawAgentLoop`，实现 `JiuwenClawAgentLoop`，只替换 runtime-specific 部分。
2. 初始阶段保持 `PinchBenchRewardManager` 不变。
3. 先用一个简单 task 做 smoke test：`BATCH_SIZE=1`，`MAX_TURNS=3`。
4. 检查 `AgentLoopOutput.extra_fields` 是否包含 `tool_rewards`、`turn_scores`、`trajectory`、`terminal_success`、`task_id`。
5. 短 LoRA 训练前后各跑一次 RL8 benchmark。
6. 训练闭环稳定后，再优化 PRM 质量或数据构造。
