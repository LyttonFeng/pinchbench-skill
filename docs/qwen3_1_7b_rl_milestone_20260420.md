# Qwen3-1.7B RL Milestone

日期：2026-04-20

## 结论

这次 milestone 跑通了 `Qwen/Qwen3-1.7B` 的 non-think 在线 RL 训练闭环：

- 训练框架：`veRL 0.7.1 + REINFORCE++ + LoRA`
- Runtime：OpenClaw runtime 跑在 ECS
- 训练模型：`Qwen/Qwen3-1.7B`
- Reward：`oracle-judge`，PRM judge 使用 DashScope `qwen-plus`
- 训练步数：`8 steps`
- Checkpoint：成功保存 `global_step_4` 和 `global_step_8` 的 LoRA adapter
- Benchmark：`global_step_8` 完整跑完 RL8，得分 `2.99/8 = 37.4%`

这轮的工程意义大于效果意义：训练、LoRA 同步、LoRA-only checkpoint、vLLM LoRA serving、本地 Mac RL8 benchmark 全链路跑通。效果上，`global_step_8` 相比 non-think baseline 只有小幅提升，弱于 Qwen3-1.7B think baseline，不适合作为最终宣传结果。

## 机器与服务

### RunPod

训练和推理服务使用同一台 RunPod：

```bash
ssh root@195.26.232.162 -p 28610 -i ~/.ssh/id_ed25519
```

RunPod 负责：

- veRL 训练
- vLLM 加载 base model / LoRA adapter
- GPU 推理

### ECS OpenClaw Runtime

训练 rollout 期间，OpenClaw runtime 跑在 ECS：

```bash
export OPENCLAW_HOST=8.163.82.224
export OPENCLAW_PORT=22
export OPENCLAW_USER=root
```

Benchmark 不在 RunPod 跑，在本地 Mac 跑。RunPod 只负责模型推理。

## 代码状态

本次 milestone 依赖以下关键文件：

| 文件 | 作用 |
|---|---|
| `rl/train/run_reinforce_lora.sh` | veRL 训练入口，封装 Hydra overrides、LoRA、reward、runtime 环境变量 |
| `rl/train/launch_main_ppo.py` | 训练启动 wrapper，预加载项目 patch |
| `rl/agent_loop/openclaw_agent_loop.py` | OpenClaw runtime adapter，负责多轮 tool rollout、FIFO、response compaction、reward 对齐 |
| `rl/agent_loop/reward.py` | process reward / PRM judge 调用逻辑 |
| `rl/train/reward_manager.py` | veRL RewardManager，读取 agent loop 输出的 token reward |
| `rl/verl_lora_only_ckpt_patch.py` | 保存 LoRA-only checkpoint |
| `rl/verl_vllm_lora_empty_guard_patch.py` | LoRA 同步诊断与空 LoRA hard fail |
| `rl/transformers_qwen3_5_patch.py` | Qwen3.5 兼容 patch，本实验不是主路径但会随训练 wrapper 加载 |
| `rl/verl_qwen3_5_generation_patch.py` | Qwen3.5 generation config 兼容 patch |
| `scripts/run_bench_rl8.sh` | 本地 Mac 跑 RL8 benchmark |
| `scripts/run_bench_rl8_lora.sh` | 本地 Mac 跑 LoRA RL8 benchmark |
| `rl/scripts/start_vllm.sh` | RunPod 上启动 vLLM / LoRA serving |

## 训练配置

最终跑通的配置如下。

```bash
export VERL_MODEL=Qwen/Qwen3-1.7B
export REWARD_MODE=oracle-judge
export PRM_MODEL=qwen-plus
export OPENCLAW_MODEL_REASONING=0
export MAX_TURNS=10
export ROLLOUT_LAYERED_SUMMON=False
export BATCH_SIZE=1
export MICRO_BATCH=1
export MAX_RESPONSE_LENGTH=8192
export VLLM_MAX_MODEL_LEN=28672
export VLLM_GPU_MEM_UTIL=0.22
export SAVE_FREQ=4
export TOTAL_TRAINING_STEPS=8
export RUN_VERSION=qwen31_nonthink_bt1_turn10_oracle_prm_qwenplus_resp8192_save4
export OPENCLAW_HOST=8.163.82.224
export OPENCLAW_PORT=22
export OPENCLAW_USER=root
```

启动命令：

```bash
cd /workspace/pinchbench-skill
source ~/.pinchbench_env 2>/dev/null || true
bash rl/train/run_reinforce_lora.sh 2>&1 \
  | tee /tmp/train_qwen31_nonthink_bt1_turn10_oracle_prm_qwenplus_resp8192_save4.log
```

后台启动时使用过：

```bash
cd /workspace/pinchbench-skill
nohup bash -lc '
source ~/.pinchbench_env 2>/dev/null || true
export VERL_MODEL=Qwen/Qwen3-1.7B
export REWARD_MODE=oracle-judge
export PRM_MODEL=qwen-plus
export OPENCLAW_MODEL_REASONING=0
export MAX_TURNS=10
export ROLLOUT_LAYERED_SUMMON=False
export BATCH_SIZE=1
export MICRO_BATCH=1
export MAX_RESPONSE_LENGTH=8192
export VLLM_MAX_MODEL_LEN=28672
export VLLM_GPU_MEM_UTIL=0.22
export SAVE_FREQ=4
export TOTAL_TRAINING_STEPS=8
export RUN_VERSION=qwen31_nonthink_bt1_turn10_oracle_prm_qwenplus_resp8192_save4
export OPENCLAW_HOST=8.163.82.224
export OPENCLAW_PORT=22
export OPENCLAW_USER=root
bash rl/train/run_reinforce_lora.sh 2>&1 \
  | tee /tmp/train_qwen31_nonthink_bt1_turn10_oracle_prm_qwenplus_resp8192_save4.log
' >/tmp/train_qwen31_nonthink_bt1_turn10_oracle_prm_qwenplus_resp8192_save4.start.log 2>&1 &
```

## 训练日志

训练日志：

```bash
/tmp/train_qwen31_nonthink_bt1_turn10_oracle_prm_qwenplus_resp8192_save4.log
```

看训练进度：

```bash
tail -f /tmp/train_qwen31_nonthink_bt1_turn10_oracle_prm_qwenplus_resp8192_save4.log
```

过滤关键信息：

```bash
grep -E "global_step|\\[PRM\\]|Reward:|collect_lora_params done|saved .*LoRA|Traceback|OOM|ERROR" \
  /tmp/train_qwen31_nonthink_bt1_turn10_oracle_prm_qwenplus_resp8192_save4.log
```

## 关键训练结果

训练完成 `8/8`：

```text
Training Progress: 100%|██████████| 8/8
training/global_step:8
```

保存了两个 LoRA checkpoint：

```bash
/workspace/pinchbench-skill/rl/checkpoints/reinforce_lora_qwen31_nonthink_bt1_turn10_oracle_prm_qwenplus_resp8192_save4/global_step_4/actor/lora_adapter

/workspace/pinchbench-skill/rl/checkpoints/reinforce_lora_qwen31_nonthink_bt1_turn10_oracle_prm_qwenplus_resp8192_save4/global_step_8/actor/lora_adapter
```

文件大小：

```text
global_step_4/actor/lora_adapter/adapter_model.safetensors 139512944 bytes
global_step_4/actor/lora_adapter/adapter_config.json 1422 bytes
global_step_8/actor/lora_adapter/adapter_model.safetensors 139512944 bytes
global_step_8/actor/lora_adapter/adapter_config.json 1422 bytes
```

LoRA 同步健康信号：

```text
collect_lora_params done count=392
[pinchbench_lora_only_ckpt] saved 392 LoRA tensors
```

`count=392` 是本实验中 Qwen3-1.7B LoRA 同步正常的关键检查点。出现 `count=0` 说明 LoRA 没有同步成功，训练无效。

最终 validation：

```text
val-core/pinchbench/reward/mean@1: -0.07687864289619029
val-aux/num_turns/min: 3
val-aux/num_turns/max: 5
val-aux/num_turns/mean: 3.5
```

## Reward 与 Advantage 观察

训练期间 reward / advantage 数值是健康的：

- 没有 NaN / inf
- advantage 不是全 0
- `critic/advantages/mean` 接近 0 是正常的，因为 REINFORCE++ 会中心化/归一化
- `grad_norm` 稳定，没有梯度爆炸
- KL 没有明显跑飞

典型 step：

```text
step 1: reward_mean=-0.50, adv_min=-0.498, adv_max=0.0017, grad_norm=0.0227
step 2: reward_mean=-1.40, adv_min=-1.399, adv_max=0.0017, grad_norm=0.0199
step 5: reward_mean=-9.20, adv_min=-8.980, adv_max=0.0031, grad_norm=0.0071
step 6: reward_mean=+0.83, adv_min=-0.003, adv_max=0.8540, grad_norm=0.0202
step 8: reward_mean=-0.38, adv_min=-0.350, adv_max=0.0019, grad_norm=0.0271
```

`step 5` 有明显大负 outlier，原因是模型在某些任务里反复重复无效工具动作，PRM 多 turn 连续扣分。这不是数值爆炸，但长训时建议加入 reward clip。

推荐后续长训配置：

```bash
REWARD_CLIP_MIN=-3
REWARD_CLIP_MAX=3
```

或者在 reward manager / agent loop 内对 process reward 做 per-episode clipping，防止单条长坏轨迹主导一次 update。

## FIFO 与 Response Budget

本轮使用：

```bash
MAX_PROMPT_LENGTH=20000
MAX_RESPONSE_LENGTH=8192
VLLM_MAX_MODEL_LEN=28672
```

三者含义不同：

- `MAX_PROMPT_LENGTH` / `PINCHBENCH_AGENT_MAX_PROMPT_TOKENS`：控制每次调用模型时的 input/history FIFO。
- `MAX_RESPONSE_LENGTH`：控制整条 episode 最终进入训练的 response token budget。
- `VLLM_MAX_MODEL_LEN`：控制 vLLM 单次生成的最大上下文窗口。

当前 response 超限处理不是简单截尾，而是 turn-aligned compaction：

- 如果生成前 `all_response_ids >= MAX_RESPONSE_LENGTH`，结束 episode。
- 如果某一轮生成后超过 budget，丢最旧的完整 turn，保留最新 turns。
- 如果最新单个 turn 就超过 budget，只保留最后 `MAX_RESPONSE_LENGTH` 个 tail tokens。

因此，FIFO 只控制“每次给模型看的输入长度”，不能自动控制“整条 episode 训练 token 总量”。`MAX_RESPONSE_LENGTH=8192` 是为了控制 actor update 的反传显存。

## OOM 与修复

之前用：

```bash
MAX_RESPONSE_LENGTH=12000
VLLM_MAX_MODEL_LEN=32768
VLLM_GPU_MEM_UTIL=0.28
SAVE_FREQ=8
```

训练到 `global_step=7` 后 GPU OOM：

```text
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 8.25 GiB.
GPU total: 44.40 GiB
free: 6.11 GiB
process has 37.13 GiB in use
```

当时没有保存 checkpoint，因为 `SAVE_FREQ=8` 且崩在 step 7。

修复后的稳定配置：

```bash
MAX_RESPONSE_LENGTH=8192
VLLM_MAX_MODEL_LEN=28672
VLLM_GPU_MEM_UTIL=0.22
SAVE_FREQ=4
TOTAL_TRAINING_STEPS=8
```

这组配置成功完成 8 step，并保存 `global_step_4` / `global_step_8`。

## PRM Judge

本轮使用：

```bash
REWARD_MODE=oracle-judge
PRM_MODEL=qwen-plus
```

PRM 请求必须走 DashScope：

```text
[PRM] Calling judge for turn 0, url=https://dashscope.aliyuncs.com/compatible-mode/v1, model=qwen-plus
```

踩坑：之前 `openclaw_agent_loop.py` 会优先把 PRM base URL 指向 rollout vLLM 动态地址，导致 `qwen-plus` 请求被发到本地 vLLM，出现：

```text
The model `qwen-plus` does not exist
```

修复方式：

- `self-judge` 才使用 rollout vLLM 动态 URL。
- `oracle-judge` 固定使用 `PRM_VLLM_BASE_URL` / DashScope URL。

本轮训练验证 PRM 路由已正确。

## vLLM LoRA Serving

训练后用 `global_step_8` 跑 RL8 benchmark，需要先在 RunPod 上启动 LoRA vLLM：

```bash
cd /workspace/pinchbench-skill
source ~/.pinchbench_env 2>/dev/null || true

export LORA_ADAPTER_PATH=/workspace/pinchbench-skill/rl/checkpoints/reinforce_lora_qwen31_nonthink_bt1_turn10_oracle_prm_qwenplus_resp8192_save4/global_step_8/actor/lora_adapter
export VLLM_LORA_NAME=pinchbench-lora-qwen31-step8
export VLLM_PORT=8024
export VLLM_GPU_MEM_UTIL=0.70
export LORA_RANK=32

bash rl/scripts/start_vllm.sh Qwen/Qwen3-1.7B \
  > /tmp/vllm_qwen31_lora_step8_8024.log 2>&1 &
```

检查服务：

```bash
curl -sSf http://127.0.0.1:8024/v1/models
```

应看到：

```text
Qwen/Qwen3-1.7B
Qwen3-1.7B
pinchbench-lora-qwen31-step8
```

注意：`rl/scripts/start_vllm.sh` 当前默认：

```bash
--enable-auto-tool-choice
--tool-call-parser hermes
--reasoning-parser deepseek_r1
```

本轮 benchmark 走 non-think OpenClaw 配置：

```bash
OPENCLAW_MODEL_REASONING=0
```

## 本地 Mac RL8 Benchmark

本地 Mac 建 tunnel：

```bash
ssh -N \
  -o ServerAliveInterval=30 \
  -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18024:127.0.0.1:8024 \
  root@195.26.232.162 -p 28610 -i ~/.ssh/id_ed25519
```

检查 tunnel：

```bash
curl -sSf http://127.0.0.1:18024/v1/models
```

跑 RL8：

```bash
cd /Users/lytton/work/reinforement_learning/pinchbench-skill
source ~/.pinchbench_env

OPENCLAW_MODEL_REASONING=0 \
MODEL=pinchbench-lora-qwen31-step8 \
BASE_URL=http://127.0.0.1:18024/v1 \
SAVE_RL8_COMPARE=1 \
RL8_COMPARE_PREFIX=lora_qwen31_step8_resp8192_save4 \
bash scripts/run_bench_rl8_lora.sh
```

结果文件：

```text
results/0119_pinchbench-lora-qwen31-step8.json
results/compare/lora_qwen31_step8_resp8192_save4.json
results/0119_transcripts/
```

## RL8 Benchmark 结果

`global_step_8` RL8 总分：

```text
2.99 / 8
37.4%
```

单项：

| Task | Score |
|---|---:|
| `task_02_stock` | `0.0000` |
| `task_10_workflow` | `0.5333` |
| `task_12_skill_search` | `0.0000` |
| `task_16_email_triage` | `0.3600` |
| `task_18_market_research` | `0.7412` |
| `task_18_spreadsheet_summary` | `0.0250` |
| `task_22_second_brain` | `1.0000` |
| `task_24_polymarket_briefing` | `0.3333` |

对照已知 baseline：

| 设置 | RL8 |
|---|---:|
| Qwen3-1.7B non-think baseline | `2.76/8` |
| Qwen3-1.7B think baseline | `4.24/8` |
| Qwen3-1.7B non-think RL global_step_8 | `2.99/8` |

结论：

- 相比 non-think baseline 小幅提升：`2.76 -> 2.99`
- 明显弱于 think baseline：`4.24`
- `task_22_second_brain` 有明显成功信号：`1.0`
- `task_18_market_research` 较好：`0.7412`
- `task_18_spreadsheet_summary` 基本未学会：`0.025`
- `task_02_stock` / `task_12_skill_search` 失败，说明工具执行和精确文件修改能力仍弱

这轮不适合作为“显著提升”的宣传结果，但适合作为“Qwen3-1.7B 在线 RL 工程闭环跑通”的 milestone。

## 重要踩坑

### 1. `ROLLOUT_LAYERED_SUMMON=False`

Qwen3-1.7B 训练必须设置：

```bash
ROLLOUT_LAYERED_SUMMON=False
```

否则 veRL/FSDP/PEFT 的 LoRA 参数收集可能为空，出现：

```text
collect_lora_params done count=0
empty LoRA model reached vLLM merge path
```

正确健康信号：

```text
collect_lora_params done count=392
```

解决过程：

```text
1. 初始现象：vLLM update_weights 阶段报 StopIteration。
2. 表层堆栈：vLLM add_lora() -> LoRAModelManager._create_merged_loras_inplace()。
3. 直接原因：vLLM 收到的 LoRA request 里没有任何 LoRA layer。
4. 诊断补丁：rl/verl_vllm_lora_empty_guard_patch.py 打印 count/sample，并在空 LoRA 时 hard fail。
5. 根因定位：layered_summon=True 时，veRL 的 layered_summon_lora_params() 对当前 Qwen3-1.7B FSDP/PEFT 命名收集不到 LoRA。
6. 最终修复：ROLLOUT_LAYERED_SUMMON=False，改走普通 FSDP.summon_full_params + get_peft_model_state_dict()。
```

关键日志对比：

```text
# 错误
[vllm_lora_debug] vLLM _update_weights count=0 base_sync_done=True has_peft_config=True sample=[]
empty LoRA model reached vLLM merge path

# 正确
collect_lora_params done count=392
```

### 2. 不允许静默空 LoRA

`verl_vllm_lora_empty_guard_patch.py` 会在空 LoRA 时 hard fail，这是有意的。不要把空 LoRA 跳过继续训练，否则训练看似在跑，实际 policy 没有更新。

### 3. LoRA-only checkpoint 保存路径要匹配 `layered_summon`

之前保存阶段曾经写出 16 bytes 的空 `adapter_model.safetensors`。根因是训练同步路径用 `layered_summon=False` 能拿到 LoRA，但保存 patch 固定用了 layered summon，导致保存为空。

修复后保存日志应看到：

```text
[pinchbench_lora_only_ckpt] saved 392 LoRA tensors
```

并且 adapter 文件大小约：

```text
adapter_model.safetensors 139512944 bytes
```

如果看到 `adapter_model.safetensors` 只有十几 bytes，基本就是空 adapter，不能用于 benchmark，也不能作为训练成果。

```text
139512944 bytes
```

### 4. `qwen3.6-plus` 不可用时用 `qwen-plus`

DashScope 曾返回：

```text
The model `qwen3.6-plus` does not exist
```

本轮改用：

```bash
PRM_MODEL=qwen-plus
```

`qwen-plus` 足够做 PRM judge。

### 5. Benchmark 必须在本地 Mac 跑

不要在 RunPod 里跑 benchmark。RunPod 只负责 vLLM 推理。本地 Mac 负责：

- OpenClaw benchmark runner
- PinchBench grading
- 结果保存到本地 `results/`

### 6. SSH tunnel 在 Codex sandbox 下可能需要提权网络权限

本地连接 `127.0.0.1:18024` 在普通 sandbox 中可能报：

```text
Operation not permitted
```

这不是 vLLM 服务错误，是本地工具 sandbox 限制。使用已授权的网络命令或提权执行即可。

## 后续建议

### 短期

1. 对 `global_step_4` 也跑一次 RL8，判断 step 8 是否已经过拟合/退化。
2. 做 non-think RL vs think baseline 的同环境多次重复，避免单次 RL8 方差误导。
3. 对 `task_22_second_brain` 单独分析轨迹，确认 RL 是否真的学到 memory 写读策略。

### 中期

1. 加 reward clipping，避免 step 5 这类大负 outlier 主导更新。
2. 针对 spreadsheet 任务构造 DPO / RL 数据，不要只靠 8 条 RL8 原题。
3. 增加 “reward-bearing turn 保留” 策略，减少 response compaction 丢掉关键早期动作的问题。
4. 对 `task_02_stock` 和 `task_12_skill_search` 做专项数据，因为这两个任务当前仍然是 0。

### 长训推荐起点

```bash
VERL_MODEL=Qwen/Qwen3-1.7B
OPENCLAW_MODEL_REASONING=0
REWARD_MODE=oracle-judge
PRM_MODEL=qwen-plus
BATCH_SIZE=1
MICRO_BATCH=1
MAX_TURNS=10
MAX_RESPONSE_LENGTH=8192
VLLM_MAX_MODEL_LEN=28672
VLLM_GPU_MEM_UTIL=0.22
ROLLOUT_LAYERED_SUMMON=False
SAVE_FREQ=4
TEST_FREQ=4
```

如果继续 OOM：

```bash
MAX_RESPONSE_LENGTH=6000
VLLM_MAX_MODEL_LEN=26000
VLLM_GPU_MEM_UTIL=0.20
```

如果目标是效果而不是仅跑通，应优先尝试：

- Qwen3-1.7B think-mode 推理/训练对照
- 更强数据构造，而不是只重复 8 条 RL8
- task-specific DPO warmup + RL
