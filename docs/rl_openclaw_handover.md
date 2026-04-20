# JiuwenClaw-RL Engineering Handover

Last updated: 2026-04-20

This handover is for engineers who will continue the RL implementation in JiuwenClaw. The current prototype uses:

- `veRL` as the trainer.
- `Qwen3-1.7B` / Qwen-family models as policy models.
- `OpenClaw runtime` as the current agent runtime.
- `PinchBench` tasks and grading as the evaluation environment.

The expected next step is to keep the veRL + PinchBench parts and replace the OpenClaw runtime adapter with a JiuwenClaw runtime adapter.

## One-Line Summary

We implemented an online RL loop for a multi-turn tool agent: veRL samples a PinchBench task, launches an agent runtime, collects tool-use trajectories, grades the final workspace, converts terminal/process rewards into token-level reward tensors, and updates a LoRA policy with REINFORCE++.

## Current Architecture

```text
veRL PPO/REINFORCE++ trainer
  -> custom AgentLoop
    -> start runtime episode
       currently: OpenClaw on ECS
       next: JiuwenClaw runtime
    -> runtime sends OpenAI-compatible chat/tool requests
    -> AgentLoop calls veRL vLLM rollout server
    -> model returns text/tool calls
    -> runtime executes tools and continues
    -> AgentLoop collects response tokens + tool trajectory
  -> PinchBench grading
  -> terminal reward + process reward
  -> custom veRL RewardManager writes token-level rewards
  -> LoRA policy update
```

The current implementation is a working adapter around OpenClaw. The important reusable part is the veRL-side contract: an `AgentLoopOutput` with prompt ids, response ids, response mask/logprobs, extra trajectory fields, and token-level rewards.

## Important Files

| File | Purpose |
|---|---|
| `rl/train/run_reinforce_lora.sh` | Main training entrypoint. Sets veRL overrides, LoRA config, reward manager, OpenClaw/Jiuwen runtime env, and preflight checks. |
| `rl/train/launch_main_ppo.py` | Imports local compatibility patches before launching `verl.trainer.main_ppo`. |
| `rl/agent_loop/openclaw_agent_loop.py` | Current runtime adapter. This is the main file to replace or fork for JiuwenClaw runtime. |
| `rl/train/reward_manager.py` | Custom veRL reward manager. Converts per-turn rewards into token-level reward tensor. Keep this. |
| `rl/agent_loop/reward.py` | Terminal reward + process reward / PRM logic and task rubrics. Keep and refine. |
| `sitecustomize.py` | Auto-loads veRL/vLLM compatibility patches inside Ray workers. |
| `scripts/benchmark.py`, `scripts/lib_agent.py`, `scripts/run_bench_rl8.sh` | Local Mac benchmark path against a served model. |
| `docs/rl_openclaw_handover.md` | This document. |

## What Was Changed Around veRL

This project does not fork veRL as a large permanent patchset. The implementation is mostly done by:

- Using veRL's existing PPO entrypoint.
- Registering a custom multi-turn `AgentLoop`.
- Registering a custom `RewardManager`.
- Loading small compatibility/debug patches through `sitecustomize.py`.
- Passing all behavior through Hydra overrides in `rl/train/run_reinforce_lora.sh`.

The next engineer should treat these as the reusable veRL changes.

### 1. Algorithm Configuration: REINFORCE++ Through veRL PPO

File: `rl/train/run_reinforce_lora.sh`

We keep `verl.trainer.main_ppo` as the trainer, but configure it into a critic-free REINFORCE++ style update:

```bash
algorithm.adv_estimator=reinforce_plus_plus
algorithm.gamma=0.0
trainer.critic_warmup=0
algorithm.use_kl_in_reward=True
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

Practical meaning:

- No critic/value model is trained.
- Each PinchBench episode produces reward from the external environment.
- KL is still used to keep the LoRA policy close to the base model.
- This is simpler than GRPO for live-agent tasks because we do not need multiple sampled completions per prompt to form a group baseline. A live runtime episode is slow and stateful, so group sampling multiplies runtime cost.

### 2. Multi-Turn Runtime Integration Through AgentLoop

Files:

- `rl/agent_loop/config.yaml`
- `rl/agent_loop/openclaw_agent_loop.py`

veRL's normal rollout assumes direct model generation. We use veRL's experimental multi-turn agent-loop path:

```bash
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.agent.default_agent_loop=openclaw_agent
actor_rollout_ref.rollout.agent.agent_loop_config_path=rl/agent_loop/config.yaml
```

The custom `OpenClawAgentLoop` is the bridge between veRL and the external runtime:

- veRL samples one row from `train.parquet`.
- AgentLoop starts one runtime episode.
- Runtime asks for model completions.
- AgentLoop calls veRL's vLLM rollout server through `server_manager.generate(...)`.
- Runtime executes tools.
- AgentLoop collects generated token ids, masks, logprobs, transcript, grading result, and rewards.
- AgentLoop returns `AgentLoopOutput`.

For JiuwenClaw, this file is the main replacement point. The algorithm does not require OpenClaw specifically; it requires a runtime adapter that can produce the same `AgentLoopOutput`.

### 3. Token-Level Reward Manager

File: `rl/train/reward_manager.py`

This is the core veRL-side reward change.

Why it was needed:

- veRL's default custom reward flow is mostly scalar reward per sample.
- A tool agent trajectory has many assistant turns.
- We need process reward and terminal reward to land on the tokens that caused the actions, not only the final token.

Implemented behavior:

- Define `PinchBenchRewardManager`.
- Read `extra_fields["tool_rewards"]` from `AgentLoopOutput`.
- Write token-aligned reward values into veRL's reward tensor.
- If `tool_rewards` is missing, fall back to `turn_scores`.
- If turn alignment fails, fall back to final-token scalar reward.

Expected adapter contract:

```python
extra_fields={
    "tool_rewards": [...],  # same length as response_ids
    "turn_scores": [...],   # per assistant turn
    "task_id": "...",
    "trajectory": [...],
}
```

If JiuwenClaw can provide `tool_rewards`, training keeps the intended token-level credit assignment. If it only provides a scalar score, the training path still runs but degrades to weaker final-token credit assignment.

### 4. Process Reward / PRM Logic

Files:

- `rl/agent_loop/reward.py`
- `rl/agent_loop/openclaw_agent_loop.py`

The runtime adapter computes terminal success from PinchBench grading, then calls process-reward logic per assistant turn.

Current reward components:

- `terminal_reward`: final PinchBench success/failure signal.
- `process_reward`: per-turn quality score from rule/self-judge/oracle-judge.
- `total_reward`: sum of turn rewards after terminal reward is attached.

Current modes:

```bash
REWARD_MODE=baseline
REWARD_MODE=rule
REWARD_MODE=self-judge
REWARD_MODE=oracle-judge
```

The default experimental path has been `self-judge`. For cleaner offline data production or stronger labels, use `oracle-judge` with `qwen-plus`.

### 5. Per-Task EMA Baseline

Files:

- `rl/train/reward_manager.py`
- `rl/agent_loop/openclaw_agent_loop.py`

RL8 tasks have very different difficulty. A single global reward baseline makes hard tasks look permanently bad and easy tasks look permanently good.

Implemented behavior:

- Track an EMA baseline per `task_id`.
- Normalize rewards around that task's recent mean.
- Defaults:

```bash
PINCHBENCH_TASK_EMA_ALPHA=0.1
PINCHBENCH_TASK_EMA_INIT=0.1
```

This is not a critic. It is a lightweight per-task baseline for stabilizing sparse agent rewards.

### 6. LoRA Training And Checkpoint Handling

Files:

- `rl/train/run_reinforce_lora.sh`
- `rl/verl_lora_only_ckpt_patch.py`
- `rl/verl_best_ckpt_patch.py`

The current training path is LoRA-only:

```bash
actor_rollout_ref.model.lora_rank=32
actor_rollout_ref.model.lora_alpha=64
actor_rollout_ref.model.target_modules=...
```

Checkpoint behavior added around veRL:

- Save LoRA adapter checkpoints for serving/evaluation.
- Optionally keep best checkpoint by validation reward.
- Optionally keep latest checkpoint for debugging.

This matters because the served model for PinchBench benchmark should load the LoRA adapter, not a full FSDP training checkpoint.

### 7. veRL/vLLM Compatibility Patches

Files loaded by `sitecustomize.py`:

- `rl/verl_debug_metrics_patch.py`
- `rl/verl_best_ckpt_patch.py`
- `rl/verl_lora_only_ckpt_patch.py`
- `rl/transformers_qwen3_5_patch.py`
- `rl/verl_qwen3_5_generation_patch.py`
- `rl/verl_vllm_lora_empty_guard_patch.py`

Why `sitecustomize.py` is used:

- Ray workers import Python modules in separate processes.
- Setting `PYTHONPATH` to repo root makes Python auto-load `sitecustomize.py`.
- This ensures patches are applied inside `TaskRunner`, `WorkerDict`, and vLLM server workers.

Patch intent:

- `verl_debug_metrics_patch.py`: add debug visibility when rollout/reward metrics are missing or malformed.
- `verl_best_ckpt_patch.py`: keep best checkpoint by validation reward.
- `verl_lora_only_ckpt_patch.py`: save lightweight LoRA adapter artifacts.
- `transformers_qwen3_5_patch.py`: compatibility alias for Qwen3.5 config classes.
- `verl_qwen3_5_generation_patch.py`: handle Qwen3.5 models without `generation_config.json`.
- `verl_vllm_lora_empty_guard_patch.py`: fail loudly if vLLM receives an empty LoRA during weight sync.

Important distinction:

- The algorithmic changes are AgentLoop, RewardManager, PRM, and REINFORCE++ config.
- The Qwen3.5 patches are compatibility workarounds.
- The empty-LoRA guard is a debug/safety patch, not a fix. If it fires, training is invalid because no LoRA tensors reached vLLM.

### 8. Current Known veRL Issues

Qwen3-1.7B:

- Works better than Qwen3.5-2B as the current training target.
- `ROLLOUT_LAYERED_SUMMON=False` is required in the current stack.
- Healthy LoRA sync should show:

```text
collect_lora_params done count=392
```

Qwen3.5-2B:

- Hit multiple veRL/vLLM/Transformers compatibility issues.
- `generation_config.json` missing path needed a patch.
- vLLM Qwen3.5 config handling incorrectly touched VL paths in some versions.
- Not recommended as the immediate engineering baseline.

Long think-mode trajectories:

- Prompt FIFO prevents context overflow.
- Backward can still OOM if the generated response sequence is too long.
- Latest Qwen3-1.7B think run reached `global_step=6` with `MAX_TURNS=10`, then OOMed during backward.
- Recommended next setting is `MAX_TURNS=8`, and possibly lower `MAX_RESPONSE_LENGTH`.

## veRL Training Setup

The training script uses veRL 0.7.1 style PPO entrypoint, but configures the algorithm as REINFORCE++:

```bash
algorithm.adv_estimator=reinforce_plus_plus
algorithm.gamma=0.0
algorithm.use_kl_in_reward=True
algorithm.norm_adv_by_std_in_grpo=False
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

LoRA config:

```bash
actor_rollout_ref.model.lora_rank=32
actor_rollout_ref.model.lora_alpha=64
actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_qkvz,in_proj_ba,in_proj_a,in_proj_b,in_proj_z,out_proj]
```

Rollout config:

```bash
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.agent.default_agent_loop=openclaw_agent
actor_rollout_ref.rollout.agent.agent_loop_config_path=rl/agent_loop/config.yaml
```

For JiuwenClaw, the plan is:

- Keep veRL trainer and overrides.
- Add a new agent loop, for example `jiuwenclaw_agent_loop.py`.
- Change `default_agent_loop` from `openclaw_agent` to the new JiuwenClaw loop.
- Keep the reward manager and PinchBench grading contract.

## AgentLoop Contract

The custom agent loop must return a veRL `AgentLoopOutput`:

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

Critical fields:

- `response_ids`: flattened generated model tokens.
- `response_mask`: `1` for model-generated tokens, `0` for environment/tool tokens.
- `response_logprobs`: model token logprobs; env/tool tokens can be `0`.
- `tool_rewards`: token-aligned reward vector, same response length.
- `turn_scores`: fallback per-turn rewards.
- `trajectory`: assistant/tool transcript for PRM and debugging.

The JiuwenClaw runtime adapter must preserve this contract even if the runtime interface differs from OpenClaw.

## Runtime Adapter Responsibilities

The current OpenClaw adapter in `rl/agent_loop/openclaw_agent_loop.py` does five jobs:

1. Start one runtime episode for a PinchBench task.
2. Route runtime model requests to veRL vLLM rollout server.
3. Parse model outputs into runtime-compatible text/tool calls.
4. Collect transcript, response tokens, tool results, and workspace state.
5. Run PinchBench grading and attach reward metadata.

For JiuwenClaw, replace these OpenClaw-specific parts:

- Starting OpenClaw through SSH.
- OpenClaw agent config generation.
- OpenClaw model provider registration.
- OpenClaw transcript parsing.
- OpenClaw workspace sync.
- OpenClaw skill preflight.

Keep these generic parts:

- Chat template application.
- vLLM `server_manager.generate(...)` call.
- Prompt FIFO / context compaction.
- Response token/mask/logprob collection.
- Reward computation interface.
- `AgentLoopOutput` format.

## Prompt / Context Handling

Multi-turn agent prompts can exceed context length. The current adapter has FIFO compaction:

- `PINCHBENCH_AGENT_MAX_PROMPT_TOKENS` defaults to `MAX_PROMPT_LENGTH`.
- `_compact_messages_by_turn()` drops oldest complete assistant turns until prompt fits.
- This fixed prompt crashes such as:

```text
Prompt length exceeds the model's maximum context length
```

Important distinction:

- Prompt FIFO prevents prompt overflow.
- It does not solve backward OOM caused by too many generated response tokens.

For JiuwenClaw, keep the same strategy:

- Bound prompt tokens before calling `server_manager.generate`.
- Bound max turns.
- Track response budget and compact/stop if response tokens exceed budget.

## Reward System

Reward has two layers:

1. Terminal reward from PinchBench grading.
2. Process reward per assistant turn.

### Terminal Reward

PinchBench grading checks final output/workspace. The current code converts grading to:

- success: `+1.0 * PINCHBENCH_TERMINAL_REWARD_WEIGHT`
- failure: `0.0`
- selected "claimed done but wrote no file" cases: `-1.0`

File-creation failure penalties are handled in `rl/agent_loop/reward.py`.

### Process Reward

Process reward is produced per assistant turn:

- `baseline`: no process reward.
- `rule`: rule-based reward.
- `self-judge`: same local model judges its own turn.
- `oracle-judge`: stronger external judge, usually `qwen-plus`.

Current default:

```bash
REWARD_MODE=self-judge
```

The PRM prompt includes:

- task goal
- optional hints
- common mistakes
- previous actions
- current tool call
- current tool result preview

Score range:

```text
-0.5 to +0.2 per turn
```

Terminal reward is added to the last turn.

## Token-Level Reward Alignment

The most important veRL change is `PinchBenchRewardManager` in `rl/train/reward_manager.py`.

Why it exists:

- veRL's standard custom reward path puts one scalar on the final token.
- Multi-turn agent RL needs credit assignment across turns.

What it does:

- Creates a zero reward tensor shaped like `responses`.
- Preferentially reads `extra_fields["tool_rewards"]`.
- Writes those values directly into the reward tensor.
- Falls back to assigning `turn_scores` at `<|im_end|>` token positions.
- Falls back again to scalar reward on final valid token.

This means the runtime adapter should provide `tool_rewards` whenever possible.

## Per-Task EMA Baseline

Implemented in `rl/train/reward_manager.py`.

Problem:

- RL8 tasks have different base difficulty.
- A global reward baseline makes hard tasks poison easier tasks.

Solution:

- Maintain EMA baseline per `task_id`.
- Normalize reward as:

```text
advantage-like score = raw_reward - EMA(task_id)
```

Key env vars:

```bash
PINCHBENCH_TASK_EMA_ALPHA=0.1
PINCHBENCH_TASK_EMA_INIT=0.1
```

This is important for multi-task agent training and should remain in the JiuwenClaw version.

## PinchBench Integration

PinchBench is used in two places:

1. Training prompt data:
   - `rl/data/prompts/train.parquet`
   - `rl/data/prompts/val.parquet`

2. Grading:
   - final workspace/transcript is evaluated by PinchBench grading logic.
   - benchmark results are run from Mac with `scripts/run_bench_rl8.sh`.

The training script runs a train-vs-benchmark prompt parity check:

```bash
python3 rl/scripts/check_train_infer_parity.py
```

Keep this check. It prevents training on prompts that differ from benchmark prompts.

## Benchmark Flow

Benchmark is not run inside RunPod. Run it from the local Mac.

Typical flow:

```bash
# 1. Serve model or LoRA on RunPod
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --port 8021 \
  --max-model-len 40960 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# 2. Tunnel from Mac to RunPod
ssh -N \
  -L 127.0.0.1:18021:127.0.0.1:8021 \
  root@<pod-ip> -p <pod-port> -i ~/.ssh/id_ed25519

# 3. Run RL8 benchmark on Mac
source ~/.pinchbench_env
MODEL=Qwen3-1.7B BASE_URL=http://127.0.0.1:18021/v1 bash scripts/run_bench_rl8.sh
```

Tool parser matters:

- Qwen3 / Qwen3-1.7B: usually `hermes`.
- Qwen3.5: must use `qwen3_xml`.
- Qwen3.5-2B: avoid `--reasoning-parser deepseek_r1` with `qwen3_xml`; it can hide tool calls inside reasoning.
- Always use `--enable-auto-tool-choice`.

## Model / Experiment Status

### Qwen3-1.7B Benchmark

Observed RL8 benchmark:

| Model setting | RL8 score |
|---|---:|
| Qwen3-1.7B non-think | 34.5% |
| Qwen3-1.7B think | 53.0% |

Details:

- `docs/qwen3_1_7b_rl8_think_vs_nonthink_20260419.md`

### Qwen3-1.7B Training

The model can initialize, roll out, sync LoRA, and train for several steps.

Required setting:

```bash
ROLLOUT_LAYERED_SUMMON=False
```

Why:

- With `layered_summon=True`, veRL collected zero LoRA tensors for Qwen3-1.7B under current FSDP/PEFT naming.
- Correct sync shows:

```text
collect_lora_params done count=392
```

Current issue:

- Think-mode training OOMs during `loss.backward()` for long trajectories.
- `MAX_TURNS=16` OOMed.
- `BATCH_SIZE=1 + MAX_TURNS=10` reached `global_step=6` then OOMed.
- Latest failing log:

```text
/tmp/train_qwen31_think_16_bt1_turn10.log
```

OOM signature:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ~8.04 GiB
```

Reason:

- Think mode generates long hidden reasoning.
- Long tool trajectories produce 20k+ training sequences.
- Backward peak memory exceeds 44GB GPU.

Recommended next training attempt:

```bash
BATCH_SIZE=1
MICRO_BATCH=1
MAX_TURNS=8
MAX_RESPONSE_LENGTH=<lower than 12000 if needed>
ROLLOUT_LAYERED_SUMMON=False
OPENCLAW_MODEL_REASONING=1
```

Or run no-think first for a stable LoRA training baseline:

```bash
OPENCLAW_MODEL_REASONING=0
```

### Qwen3.5-2B

Do not continue this as the main path right now.

Problems encountered:

- Transformers/veRL/vLLM compatibility issues with `qwen3_5`.
- Missing `generation_config.json`.
- Qwen3.5 top-level config wraps `text_config`.
- vLLM LoRA sync instability.
- Host RAM pressure during initialization.

Compatibility patches exist:

- `rl/transformers_qwen3_5_patch.py`
- `rl/verl_qwen3_5_generation_patch.py`
- loaded by `sitecustomize.py`

These are debugging aids, not a fully stable production path.

## veRL / vLLM Patches

Loaded automatically through `sitecustomize.py` because Ray workers inherit `PYTHONPATH`.

Patches:

- `rl/verl_debug_metrics_patch.py`
  - Avoids crash on empty rollout probability diff metrics.
- `rl/verl_best_ckpt_patch.py`
  - Keeps best checkpoint by validation metric and optionally latest.
- `rl/verl_lora_only_ckpt_patch.py`
  - Saves smaller LoRA adapter checkpoints.
- `rl/transformers_qwen3_5_patch.py`
  - Qwen3.5 config/model compatibility shim.
- `rl/verl_qwen3_5_generation_patch.py`
  - Generation config fallback for Qwen3.5.
- `rl/verl_vllm_lora_empty_guard_patch.py`
  - Adds diagnostics and hard-fails clearly if LoRA collection is empty.

Important: `verl_vllm_lora_empty_guard_patch.py` currently fails loudly if LoRA is empty. This is intentional. Silently continuing with empty LoRA would make training meaningless.

## Known veRL Site-Package Patch

The `masked_whiten` epsilon patch is still a direct edit in installed veRL:

```python
# /usr/local/lib/python3.12/dist-packages/verl/utils/torch_functional.py
whitened = (values - mean) * torch.rsqrt(var + 1.0)
```

Without this, sparse token reward can produce huge advantage values.

This should eventually become a clean repo patch or upstream config option.

## Current Training Command Template

For Qwen3-1.7B think, safer version:

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

## What JiuwenClaw Needs To Implement

The JiuwenClaw runtime adapter should implement the same logical interface as the OpenClaw adapter:

1. Start a runtime episode for a task.
2. Feed the task prompt to JiuwenClaw.
3. Receive model requests from JiuwenClaw.
4. Convert those requests to veRL/vLLM prompt ids.
5. Send model outputs back to JiuwenClaw as text/tool calls.
6. Collect tool trajectory and workspace state.
7. Run PinchBench grading.
8. Return `AgentLoopOutput` with token ids, masks, rewards, and extra fields.

Do not rewrite veRL training first. The fastest path is to replace only the runtime adapter while keeping:

- `run_reinforce_lora.sh`
- `PinchBenchRewardManager`
- `reward.py`
- PinchBench grading
- benchmark scripts
- LoRA checkpoint patches

## Things To Watch During Training

Healthy logs:

```text
collect_lora_params done count=392
Training Progress: ...
training/global_step:N
actor/grad_norm:<finite small number>
critic/advantages/max:<finite>
Chat template done, prompt_ids=<below budget>
```

Bad logs:

```text
collect_lora_params done count=0
empty LoRA model reached vLLM merge path
Prompt length exceeds the model's maximum context length
torch.OutOfMemoryError during loss.backward
assistant content: []
tool_calls=[]
```

## Recommended Next Steps

1. Build `JiuwenClawAgentLoop` by copying the OpenClaw adapter and replacing runtime-specific parts.
2. Keep `PinchBenchRewardManager` unchanged initially.
3. Run a smoke test with one easy task and `BATCH_SIZE=1`, `MAX_TURNS=3`.
4. Verify `AgentLoopOutput.extra_fields` contains `tool_rewards`, `turn_scores`, `trajectory`, `terminal_success`, and `task_id`.
5. Run RL8 benchmark before and after a short LoRA run.
6. Only after the loop is stable, improve PRM quality or data construction.
