# PinchBench RL Training Architecture

## 系统概览

```
┌─────────────────────────── RunPod (L40S 48GB GPU) ───────────────────────────┐
│                                                                              │
│  ┌─── veRL Trainer (REINFORCE++ / LoRA) ──────────────────────────────────┐  │
│  │                                                                        │  │
│  │  TaskRunner (Ray)                                                      │  │
│  │    │                                                                   │  │
│  │    ├── WorkerDict (FSDP Actor)  ←── LoRA weights update (backward)     │  │
│  │    │     Qwen3-4B + LoRA r=16                                          │  │
│  │    │                                                                   │  │
│  │    ├── vLLMHttpServer (Ray Actor) ←── 推理引擎 (forward)               │  │
│  │    │     Qwen3-4B + LoRA                                               │  │
│  │    │     端口: Ray动态分配 (e.g. 172.21.0.2:32795)                     │  │
│  │    │     支持: /v1/completions (text completion)                        │  │
│  │    │                                                                   │  │
│  │    └── AgentLoopWorker (Ray Actor)                                     │  │
│  │          │                                                             │  │
│  │          │  ┌─────── 每个 Step (episode) 的流程 ──────────────────┐    │  │
│  │          │  │                                                     │    │  │
│  │          │  │  1. 选 task (从 train.parquet)                      │    │  │
│  │          │  │  2. 启动 ModelProxy (HTTP反向代理)                  │    │  │
│  │          │  │  3. SSH 到 ECS 启动 OpenClaw                       │    │  │
│  │          │  │  4. 多轮交互 (Turn 0..4):                          │    │  │
│  │          │  │     ├─ OpenClaw 发请求 → ModelProxy 截获            │    │  │
│  │          │  │     ├─ ModelProxy → vLLM generate() → 生成 tokens   │    │  │
│  │          │  │     ├─ tokens → SSE stream → OpenClaw 解析          │    │  │
│  │          │  │     └─ OpenClaw 执行工具 → 下一轮                   │    │  │
│  │          │  │  5. Episode 结束 → Grading (rsync + lib_grading)   │    │  │
│  │          │  │  6. Reward 计算 (PRM self-judge + rule fallback)   │    │  │
│  │          │  │  7. 返回 AgentLoopOutput (ids, rewards)            │    │  │
│  │          │  │                                                     │    │  │
│  │          │  └─────────────────────────────────────────────────────┘    │  │
│  │                                                                        │  │
│  │  Training Step:                                                        │  │
│  │    rollout → ref_log_prob → advantage → actor_update → sync_weights    │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
          │ SSH reverse tunnel                    │ rsync (grading)
          │ (ModelProxy ← OpenClaw)               │ (workspace files)
          ▼                                       ▼
┌─────────────────────── Alibaba ECS (4核8G CPU) ──────────────────────────────┐
│                                                                              │
│  OpenClaw (AI Assistant CLI)                                                 │
│    ├─ 连接 ModelProxy (SSH 反向隧道到 RunPod)                                │
│    ├─ 在 /tmp/pinchbench/task_xx/ 工作空间执行任务                           │
│    ├─ 工具: read, write, edit, exec, web_search, web_fetch                   │
│    └─ 输出: transcript (对话记录)                                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Reward 计算流程

```
Episode 结束
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  1. Grading (Terminal Reward)                        │
│     ├─ rsync: 从 ECS 拉取 workspace 文件到 RunPod   │
│     ├─ lib_grading.grade_task(): 检查文件内容        │
│     ├─ score >= 0.5 → terminal_reward = +1.0        │
│     └─ score <  0.5 → terminal_reward = -1.0        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  2. PRM Self-Judge (Process Reward) — 每个 turn     │
│                                                      │
│  对每个 assistant turn:                              │
│    ├─ 构建 PRM prompt (build_prm_prompt):            │
│    │   ├─ Task goal (来自 TASK_RUBRICS)              │
│    │   ├─ Reference path 天眼 (qwen-plus 怎么做的)   │
│    │   ├─ Common mistakes (常见错误)                  │
│    │   ├─ Agent 的历史动作                            │
│    │   └─ 当前 turn 的工具调用和结果                  │
│    │                                                  │
│    ├─ 调 vLLM /v1/completions (Qwen3-4B 自评)       │
│    │   返回 {"score": float, "reason": "..."}        │
│    │                                                  │
│    ├─ 同时算 Rule Reward (基于 rubric 的规则):       │
│    │   ├─ 有 tool call: +0.10                        │
│    │   ├─ 参数合理: +0.05                            │
│    │   ├─ 工具执行成功: +0.10                        │
│    │   ├─ 匹配天眼路径顺序: +0.10                    │
│    │   ├─ 重复同样动作: -0.15                        │
│    │   ├─ 空响应: -0.20                              │
│    │   └─ 幻觉/拒绝: -0.20                          │
│    │                                                  │
│    └─ 融合策略:                                      │
│        ├─ Judge ≠ 0 → 70% judge + 30% rule           │
│        └─ Judge = 0 → 100% rule (fallback)           │
│                                                      │
│  最终: per_turn_rewards[-1] += terminal_reward       │
│        (terminal 只加在最后一个 turn)                 │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  3. Reward Assignment                                │
│     将 per-turn rewards 放到对应 <|im_end|> token 上 │
│     其余 token 位置 reward = 0                       │
└─────────────────────────────────────────────────────┘
```

## 名词解释

| 概念 | 含义 | 例子 |
|------|------|------|
| **Step** | 一个完整训练步 = 1个 episode | Step 4: rollout→grading→reward→backward |
| **Turn** | episode 内一次 LLM 生成 | Turn 0: 模型调 read → Turn 1: 模型调 write |
| **Terminal Reward** | 任务最终成败 | grading score≥0.5 → +1.0, 否则 -1.0 |
| **Process Reward** | 每个 turn 的行为质量 | 好的 tool call → +0.2, 重复错误 → -0.3 |
| **天眼 (Reference)** | qwen-plus 的成功路径 | "1. read config → 2. write script → 3. write notes" |
| **PRM Self-Judge** | Qwen3-4B 自评自己的行为 | "correctly reads config.json" → +0.20 |
| **ModelProxy** | HTTP 反向代理 | OpenClaw → ModelProxy → vLLM generate |

## 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Qwen3-4B + LoRA r=16 | ~4B 参数，LoRA 微调 |
| 算法 | REINFORCE++ | 无 critic，用 advantage normalization |
| Batch size | 1 | 受 ECS 资源限制 |
| Max turns | 5 | 每个 episode 最多 5 次交互 |
| Agent timeout | 120s | OpenClaw 超时 |
| GPU memory | ~29GB / 48GB | L40S 显存充裕 |
| Reward mode | self-judge | PRM + rule fallback |
| KL loss | 有 (coef=0.05) | 防止偏离参考模型太远 |
| Think mode | 开启 (Qwen3 默认) | 模型会 `<think>` 后再行动 |

## Reward 信号示例

```
好的 episode (score=-0.42):
  Turn 0: read config.json     → Judge: +0.20 "correctly reads config"
  Turn 1: write script.py      → Judge: +0.20 "following reference path"
  Turn 2: (terminal + process) → -0.86 = process(+0.14) + terminal(-1.0)
  Total: +0.185 + 0.185 + (-0.86) = -0.42

差的 episode (score=-1.48):
  Turn 0: read config.json     → Judge: +0.20 "correctly reads"
  Turn 1: exec broken code     → Judge: -0.30 "syntax error"
  Turn 2: exec same broken     → Judge: -0.30 "repeating failed"
  Turn 3: exec same broken     → Judge: -0.30 "repeating failed"
  Turn 4: (terminal + process) → -1.165 = process(-0.165) + terminal(-1.0)
  Total: 0.185 - 0.165 - 0.165 - 0.165 + (-1.165) = -1.48

REINFORCE++ advantage:
  mean_reward ≈ -1.1
  好 episode advantage: -0.42 - (-1.1) = +0.68 (正向学习信号!)
  差 episode advantage: -1.48 - (-1.1) = -0.38 (远离这个行为)
```
