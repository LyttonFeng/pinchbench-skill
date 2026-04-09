# PinchBench RL 训练架构文档

> 最后更新: 2026-04-09

---

## 1. 整体架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RunPod (L4 GPU, 24GB VRAM)                    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │                    veRL Trainer (主控)                       │     │
│  │                                                             │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │     │
│  │  │   Actor       │  │  Ref Model   │  │  Reward Manager  │  │     │
│  │  │ Qwen3-4B      │  │  Qwen3-4B    │  │  PinchBench      │  │     │
│  │  │ + LoRA (r=32) │  │  (frozen)    │  │  RewardManager   │  │     │
│  │  │ FSDP          │  │  CPU offload │  │                  │  │     │
│  │  └──────┬───────┘  └──────────────┘  └────────┬─────────┘  │     │
│  │         │                                      │            │     │
│  │         │ LoRA 权重同步 (layered_summon)        │            │     │
│  │         ▼                                      │            │     │
│  │  ┌──────────────────────────────────┐          │            │     │
│  │  │     vLLM Rollout Engine          │          │            │     │
│  │  │     (GPU 45%, TP=1)              │          │            │     │
│  │  │                                  │◄─────────┘            │     │
│  │  │  ┌────────────┐ ┌────────────┐   │  PRM self-judge      │     │
│  │  │  │ Agent 推理  │ │ PRM 推理   │   │  (同一个 vLLM)       │     │
│  │  │  │ (做任务)    │ │ (打分)     │   │                      │     │
│  │  │  └─────┬──────┘ └────────────┘   │                      │     │
│  │  └────────┼─────────────────────────┘                      │     │
│  │           │                                                 │     │
│  │  ┌────────┼─────────────────────────────────────────┐      │     │
│  │  │  OpenClawAgentLoop                                │      │     │
│  │  │        │                                          │      │     │
│  │  │  ┌─────▼──────────┐                               │      │     │
│  │  │  │  ModelProxy     │ ◄──── HTTP POST ────┐        │      │     │
│  │  │  │  (aiohttp,      │      /v1/chat/       │        │      │     │
│  │  │  │   临时端口)      │      completions     │        │      │     │
│  │  │  └────────────────┘                       │        │      │     │
│  │  └───────────────────────────────────────────┼────────┘      │     │
│  └──────────────────────────────────────────────┼───────────────┘     │
│                                                 │                     │
└─────────────────────────────────────────────────┼─────────────────────┘
                                                  │ SSH + HTTP
                                                  │
┌─────────────────────────────────────────────────┼─────────────────────┐
│                    阿里云 ECS (4核 8G)           │                     │
│                                                  │                     │
│  ┌───────────────────────────────────────────────┼──────────────┐     │
│  │              OpenClaw Runtime                  │              │     │
│  │                                                │              │     │
│  │  ┌────────────────┐     ┌─────────────────────┘───────┐      │     │
│  │  │  Gateway        │     │  models.json                │      │     │
│  │  │  (:18789)       │     │  baseUrl → ModelProxy       │      │     │
│  │  │                 │     │  (RunPod 公网 IP:PORT)      │      │     │
│  │  └────────┬───────┘     └──────────────────────────────┘      │     │
│  │           │                                                    │     │
│  │  ┌────────▼───────────────────────────────────────────┐       │     │
│  │  │              工具执行层                              │       │     │
│  │  │                                                     │       │     │
│  │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐ │       │     │
│  │  │  │ bash │ │ read │ │write │ │ edit │ │web_search│ │       │     │
│  │  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────────┘ │       │     │
│  │  │  ┌───────────┐ ┌───────────────────────────────┐   │       │     │
│  │  │  │ web_fetch │ │ exec (Python, awk, etc.)      │   │       │     │
│  │  │  └───────────┘ └───────────────────────────────┘   │       │     │
│  │  └────────────────────────────────────────────────────┘       │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │              PinchBench Grading                                │     │
│  │                                                                │     │
│  │  ┌──────────────┐  ┌────────────────────────────────────┐     │     │
│  │  │  automated    │  │  llm_judge                         │     │     │
│  │  │  (Python 脚本 │  │  POST → DashScope API              │     │     │
│  │  │   本地执行)   │  │  model: qwen-plus                  │     │     │
│  │  └──────────────┘  └────────────────────────────────────┘     │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘

外部 API:
┌─────────────────────────────────────────────┐
│  DashScope (dashscope.aliyuncs.com)          │
│  qwen-plus: Terminal Grading Judge           │
│  (仅 llm_judge / hybrid 类型的 task 使用)    │
└─────────────────────────────────────────────┘
```

---

## 2. 数据流：一个 Episode 的完整生命周期

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     veRL Trainer                             │
                    │                                                             │
Step 1              │  Dataset: 8 个 task prompt (parquet)                        │
从数据集采样 ───────│──► 随机采样 1 个 task_id + task_prompt                      │
                    │                                                             │
Step 2              │  OpenClawAgentLoop.run(task_id, task_prompt)                │
启动 episode ───────│──► 启动 ModelProxy (localhost:PORT)                         │
                    │──► SSH 到 ECS, 运行 openclaw agent --local                 │
                    │                                                             │
Step 3              │         ┌──────────────── 交互循环 ────────────────┐        │
多轮交互 ──────────│         │                                          │        │
                    │         │  OpenClaw 需要 LLM 响应                  │        │
                    │         │    │                                     │        │
                    │         │    ▼                                     │        │
                    │         │  POST /v1/chat/completions → ModelProxy  │        │
                    │         │    │                                     │        │
                    │         │    ▼                                     │        │
                    │         │  ModelProxy 把 messages 放入 Queue       │        │
                    │         │    │                                     │        │
                    │         │    ▼                                     │        │
                    │         │  AgentLoop 从 Queue 取出 request         │        │
                    │         │    │                                     │        │
                    │         │    ▼                                     │        │
                    │         │  tokenizer.apply_chat_template(messages) │        │
                    │         │    │                                     │        │
                    │         │    ▼                                     │        │
                    │         │  vLLM generate(prompt_ids, sampling_params)      │
                    │         │    │                                     │        │
                    │         │    ▼                                     │        │
                    │         │  记录 TurnRecord:                        │        │
                    │         │    {prompt_ids, response_ids, logprobs}  │        │
                    │         │    │                                     │        │
                    │         │    ▼                                     │        │
                    │         │  ModelProxy 返回 OpenAI JSON 给 OpenClaw │        │
                    │         │    │                                     │        │
                    │         │    ▼                                     │        │
                    │         │  OpenClaw 执行工具 (在 ECS 本地)         │        │
                    │         │    │                                     │        │
                    │         │    └──── 继续下一轮 ─────────────────────┘        │
                    │                                                             │
Step 4              │  OpenClaw 退出 → 加载 transcript JSONL                     │
收集结果 ──────────│                                                             │
                    │                                                             │
Step 5              │  PinchBench grading:                                        │
Terminal grading ──│    automated → 本地 Python 检查 → score                    │
                    │    llm_judge → qwen-plus API → score                       │
                    │    → terminal_success = (score >= 0.5)                      │
                    │    → terminal_reward = +1 或 -1                             │
                    │                                                             │
Step 6              │  Per-turn Process Reward (Self-Judge):                      │
Process reward ────│    for each assistant turn k:                               │
                    │      构造 PRM prompt (rubric + 天眼 + 当前 turn)            │
                    │      → vLLM generate (同一个 Qwen3-4B)                     │
                    │      → parse JSON → score ∈ [-0.5, +0.3]                  │
                    │                                                             │
Step 7              │  TrajectoryReconstructor:                                   │
轨迹重建 ──────────│    TurnRecords → AlignedTrajectory                          │
                    │    {prompt_ids, response_ids, response_mask, logprobs}      │
                    │    mask: 1=模型生成, 0=环境/模板 token                      │
                    │                                                             │
Step 8              │  Reward 分配到 token 位置:                                  │
Reward 分配 ───────│    找 <|im_end|> 位置                                       │
                    │    reward[im_end_k] = process_reward[k]                     │
                    │    reward[last_im_end] += terminal_reward                   │
                    │                                                             │
                    │  ══► 输出: (prompt_ids, response_ids, mask, rewards)        │
                    └─────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────────────────┐
Step 9              │              GRPO Update                                    │
参数更新 ──────────│                                                             │
                    │  攒够 batch_size=8 条 episode 后:                           │
                    │    ├── 计算 advantages (直接用 reward)                      │
                    │    ├── GRPO loss (ppo_epochs=2)                             │
                    │    ├── KL penalty (coef=0.05, vs ref model)                │
                    │    └── LoRA 参数更新                                        │
                    │                                                             │
Step 10             │  LoRA Sync:                                                │
权重同步 ──────────│    新 LoRA 权重 → vLLM (layered_summon, 只传 delta)         │
                    │    下一个 batch 用更新后的策略                               │
                    │                                                             │
                    └─────── 回到 Step 1, 下一个 training step ──────────────────┘
```

---

## 3. 核心组件详解

### 3.1 ModelProxy（模型代理）

```
文件: rl/agent_loop/model_proxy.py
协议: OpenAI Chat Completions API (兼容)

┌─────────────┐     HTTP POST           ┌──────────────┐
│  OpenClaw    │ ──────────────────────► │  ModelProxy   │
│  (ECS)       │  /v1/chat/completions   │  (aiohttp)    │
│              │ ◄────────────────────── │              │
│              │  OpenAI JSON response    │              │
└─────────────┘                          └──────┬───────┘
                                                │
                                         asyncio.Queue
                                                │
                                         ┌──────▼───────┐
                                         │ AgentLoop    │
                                         │              │
                                         │ get_request()│──► tokenize
                                         │              │──► vLLM generate
                                         │send_response()◄── decode
                                         └──────────────┘
```

**为什么需要 ModelProxy？**
- OpenClaw 是黑盒子进程，自己管理工具执行和对话流
- 我们只需要拦截它的 LLM 调用，替换成 veRL 的 vLLM
- ModelProxy 绑定临时端口（port=0），避免多 worker 冲突
- OpenClaw 的 models.json 里 baseUrl 指向 ModelProxy

### 3.2 TrajectoryReconstructor（轨迹重建）

```
文件: rl/agent_loop/trajectory.py

输入: [TurnRecord_1, TurnRecord_2, ..., TurnRecord_k]
      每个 TurnRecord 包含该轮的 messages, prompt_ids, response_ids, logprobs

处理:
  Turn 1: prompt_ids → initial_prompt_ids (训练时的 prompt)
           response_ids → mask=1 (模型生成)

  Turn 1→2 之间: 环境返回的 tool result tokens → mask=0

  Turn 2: response_ids → mask=1
  ...

输出: AlignedTrajectory
  ├── initial_prompt_ids: [token_ids...]     # 第一轮的 prompt
  ├── response_ids:       [token_ids...]     # 所有 turn 的 response 拼接
  ├── response_mask:      [1,1,1,0,0,1,1..] # 1=模型, 0=环境
  ├── response_logprobs:  [float...]         # mask=1 处有真实 logprob
  └── per_turn_boundaries: [(0,50),(55,120)..] # 每个 turn 在 response 中的位置
```

**为什么需要轨迹重建？**
- veRL 训练需要 token 级别的 prompt/response 分离和 mask
- OpenClaw 控制对话流，veRL 只看到一次次的 LLM 请求
- 重建过程把多轮交互"拍平"成一条 (prompt, response, mask) 序列

### 3.3 Reward 分配（Token 级别）

```
response_ids:  [t1 t2 t3 <|im_end|> env env t4 t5 <|im_end|> env t6 t7 <|im_end|>]
response_mask: [ 1  1  1     1       0   0   1  1     1       0   1  1     1      ]
reward:        [ 0  0  0   +0.15     0   0   0  0   +0.10     0   0  0  +0.00+1.0 ]
                          ↑                        ↑                      ↑
                     turn 1 reward            turn 2 reward     turn 3 + terminal
```

- Process reward 放在每个 assistant turn 的 `<|im_end|>` token 上
- Terminal reward 叠加在最后一个 turn 的 `<|im_end|>` 上
- 其余 token 的 reward = 0
- 只有 mask=1 的 token 参与梯度计算

### 3.4 Self-Judge PRM（过程奖励模型）

```
文件: rl/agent_loop/reward.py

┌──────────────────────────────────────────────────────┐
│                   PRM Prompt 构造                     │
│                                                      │
│  任务目标 ← TASK_RUBRICS[task_id]["goal"]            │
│  参考路径 ← TASK_RUBRICS[task_id]["reference_steps"] │  ← 天眼
│  常见错误 ← TASK_RUBRICS[task_id]["common_mistakes"] │  ← 天眼
│  历史行为 ← prev_turns 摘要                          │
│  当前行为 ← current_turn 的 tool + args + result     │
│                                                      │
│  → 拼接成 prompt                                     │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│             vLLM (Qwen3-4B, 同一个实例)               │
│                                                       │
│  输入: PRM prompt (~500 tokens)                       │
│  输出: {"score": 0.15, "reason": "correct web_search"}│
│                                                       │
│  temperature=0.1, max_tokens=128                      │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
                  score ∈ [-0.5, +0.3]
```

**Self-Judge 的特殊之处**：
- Agent 和 Judge 是 **同一个模型**（Qwen3-4B）
- 但 Judge 有额外信息：rubric + 天眼 reference trajectory
- Agent 不知道正确路径 → Judge 知道正确路径 → 能打出有意义的分数
- Agent 变强 → Judge 也变强 → 自进化正循环

---

## 4. GPU 显存分配

```
L4 24GB VRAM 分配:

┌─────────────────────────────────────────────────┐
│                                                 │
│  vLLM Rollout Engine        Actor Training      │
│  (gpu_memory_utilization    (剩余显存)           │
│   = 0.45)                                       │
│                                                 │
│  ┌───────────────────┐ ┌─────────────────────┐  │
│  │                   │ │                     │  │
│  │  ~10.8 GB         │ │  ~13.2 GB           │  │
│  │                   │ │                     │  │
│  │  Qwen3-4B 推理    │ │  LoRA forward       │  │
│  │  (Agent + PRM     │ │  + backward         │  │
│  │   共享)           │ │  + optimizer state   │  │
│  │                   │ │  (grad accumulation  │  │
│  │  KV cache         │ │   micro_batch=2)     │  │
│  │  prefix cache     │ │                     │  │
│  │                   │ │                     │  │
│  └───────────────────┘ └─────────────────────┘  │
│                                                 │
│  Ref Model: CPU offload (不占 GPU 显存)          │
│                                                 │
└─────────────────────────────────────────────────┘
```

**显存节省手段**：
- LoRA (rank=32): 只更新 ~2% 参数
- layered_summon: LoRA sync 只传 delta 权重
- Ref model CPU offload: 不占 GPU
- gradient_checkpointing: 用计算换显存
- micro_batch=2 + gradient accumulation: 降低峰值显存

---

## 5. 网络拓扑

```
┌─────────────┐                              ┌──────────────┐
│   RunPod     │                              │  阿里云 ECS   │
│  (GPU)       │                              │  (CPU)        │
│              │                              │              │
│  ModelProxy  │ ◄────── HTTP (公网) ──────── │  OpenClaw     │
│  :PORT       │        POST /v1/chat/...     │  Gateway      │
│              │                              │  :18789       │
│              │ ──────── SSH (公网) ────────► │              │
│              │     启动 openclaw agent       │  工具执行     │
│              │                              │  (本地文件系统│
│              │                              │   + web)      │
│  公网 IP:    │                              │              │
│  RunPod 分配 │                              │  8.163.82.224 │
└─────────────┘                              └──────────────┘
                                                    │
                                                    │ HTTPS
                                                    ▼
                                             ┌──────────────┐
                                             │  DashScope   │
                                             │  API         │
                                             │  qwen-plus   │
                                             │  (judge)     │
                                             └──────────────┘
```

**两个方向的网络连接**：
1. **RunPod → ECS (SSH)**: 启动 OpenClaw agent 子进程
2. **ECS → RunPod (HTTP)**: OpenClaw 的 LLM 请求发到 ModelProxy

**为什么 OpenClaw 在 ECS，不在 RunPod？**
- OpenClaw 需要 Node.js 运行时 + 工具环境（bash, web_search 等）
- RunPod 容器环境不一定有这些
- ECS 7x24 在线，不受 GPU 实例启停影响
- 后续可换成其他 OpenClaw 实例（只改 OPENCLAW_HOST）

---

## 6. 并发与性能

### 6.1 Episode 并发

```
一个 training step (batch_size=8):

Wave 1: 4 个 episode 并发 ──────► ~3-5 分钟
  ├── task_02 (OpenClaw session 1)
  ├── task_10 (OpenClaw session 2)
  ├── task_12 (OpenClaw session 3)
  └── task_16 (OpenClaw session 4)

Wave 2: 4 个 episode 并发 ──────► ~3-5 分钟
  ├── task_18 (OpenClaw session 5)
  ├── task_19 (OpenClaw session 6)
  ├── task_22 (OpenClaw session 7)
  └── task_24 (OpenClaw session 8)

PRM scoring: 8 × ~8 turns = ~64 次 vLLM 调用 ──► ~10 秒

GRPO update: forward + backward ──────────────► ~30 秒

总计: ~8-12 分钟 / training step
50 epochs × 8 tasks = 400 episodes ──► ~6-10 小时
```

### 6.2 瓶颈分析

| 阶段 | 耗时 | 瓶颈 |
|------|------|------|
| OpenClaw episode | 3-5 min | 工具执行 + 网络延迟 |
| PRM self-judge | ~10 sec | vLLM batch inference |
| PinchBench grading | ~5-30 sec | automated 快, llm_judge 慢 |
| GRPO update | ~30 sec | GPU 计算 |
| LoRA sync | ~2 sec | 权重传输 |

**主要瓶颈是 OpenClaw episode 执行时间**，不是 GPU 计算。

---

## 7. LoRA 权重同步流程

```
Training Step N 结束:
  Actor 参数更新 (LoRA delta 变化)
       │
       ▼
  layered_summon=True:
    只提取 LoRA adapter 权重 (几十 MB)
    而不是完整模型权重 (~8 GB)
       │
       ▼
  TensorLoRARequest → vLLM
    vLLM 在基础权重上叠加新的 LoRA adapter
       │
       ▼
  Training Step N+1:
    vLLM 用更新后的 LoRA 做推理
    Agent 行为反映最新策略
```

**为什么用 layered_summon？**
- 完整模型同步: ~8 GB, 几十秒
- LoRA delta 同步: ~50 MB, <2 秒
- 训练过程不中断 vLLM 服务

---

## 8. 容错与恢复

### 8.1 Episode 级容错

| 故障 | 处理 |
|------|------|
| OpenClaw 超时 (>600s) | kill 进程, terminal_reward=-1 |
| SSH 连接失败 | 重试 3 次, 失败则 skip |
| ModelProxy 无响应 | 超时返回错误, turn 结束 |
| vLLM OOM | 降低 batch size, 重启 |
| PRM JSON 解析失败 | fallback 返回 score=0.0 |
| Transcript 找不到 | terminal_reward=-1, 0 turns |

### 8.2 训练级容错

| 故障 | 处理 |
|------|------|
| 轨迹重建失败 (alignment mismatch) | reward=0, 不参与训练 |
| 一个 batch 全部 episode 失败 | skip 这个 step |
| ECS 断连 | 训练暂停, 等恢复 |
| RunPod 抢占 (spot) | checkpoint 恢复 (save_freq=10) |

### 8.3 Checkpoint

```
每 10 个 training step 保存:
  ├── LoRA adapter 权重
  ├── Optimizer state
  ├── Training step counter
  └── Per-task reward EMA (统计用)

恢复:
  actor_rollout_ref.model.lora_adapter_path=/path/to/checkpoint
```

---

## 9. 与 veRL 框架的集成方式

```
veRL 框架 (不修改源码)
  │
  ├── AgentLoopBase (抽象基类)
  │     └── @register("openclaw_agent")
  │           └── OpenClawAgentLoop (我们的实现)
  │
  ├── AbstractRewardManager (抽象基类)
  │     └── PinchBenchRewardManager (我们的实现)
  │
  ├── 训练配置
  │     ├── agent_loop_config_path → rl/agent_loop/config.yaml
  │     ├── reward.reward_manager.path → rl/train/reward_manager.py
  │     └── reward.reward_manager.name → PinchBenchRewardManager
  │
  └── 数据格式
        └── parquet: {prompt, extra_info, data_source, reward_model}
```

**零 veRL 改动的原因**：
- veRL 的 Agent Loop 系统支持外部注册自定义 agent loop
- Reward Manager 支持外部 Python 文件加载
- 数据格式兼容 veRL 的 parquet 标准
- 所有自定义代码在 `pinchbench-skill/rl/` 下

---

## 10. 配置速查表

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENCLAW_HOST` | localhost | ECS IP |
| `OPENCLAW_USER` | root | SSH 用户 |
| `OPENCLAW_SSH_KEY` | ~/.ssh/id_ed25519 | SSH 密钥 |
| `OPENCLAW_PORT` | 22 | SSH 端口 |
| `PINCHBENCH_DIR` | (auto) | pinchbench-skill 仓库路径 |
| `JUDGE_MODEL` | qwen-plus | Terminal grading judge |
| `JUDGE_BASE_URL` | dashscope.aliyuncs.com | Judge API 地址 |
| `JUDGE_API_KEY` | - | DashScope API key |
| `REWARD_MODE` | self-judge | PRM 模式 |
| `PRM_VLLM_BASE_URL` | localhost:8000/v1 | Self-judge vLLM 地址 |
| `PRM_MODEL` | Qwen3-4B | Self-judge 模型 |
| `BATCH_SIZE` | 8 | 每 step 的 episode 数 |
| `LORA_RANK` | 32 | LoRA rank |
| `LR` | 2e-5 | 学习率 |

### 文件清单

| 文件 | 用途 |
|------|------|
| `rl/agent_loop/openclaw_agent_loop.py` | 核心 agent loop |
| `rl/agent_loop/model_proxy.py` | HTTP 反向代理 |
| `rl/agent_loop/trajectory.py` | 轨迹重建 |
| `rl/agent_loop/reward.py` | PRM self-judge + rubric |
| `rl/agent_loop/config.yaml` | veRL 注册配置 |
| `rl/train/run_grpo_lora.sh` | 训练启动脚本 |
| `rl/train/prepare_prompts.py` | 数据准备 |
| `rl/train/reward_manager.py` | veRL reward 适配 |
| `rl/scripts/setup_ecs.sh` | ECS 初始化 |
| `rl/scripts/start_vllm.sh` | vLLM 启动 |
| `rl/judge_rubrics.md` | 评判标准 + 天眼 |
| `rl/ALGORITHM_DESIGN.md` | 算法设计文档 |
| `rl/TRAINING_ARCHITECTURE.md` | 本文档 |
