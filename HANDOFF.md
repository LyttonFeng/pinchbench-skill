# PinchBench RL — Handoff

---

## 实验结果总览

| 版本 | 总分 | 关键配置 | 备注 |
|------|------|----------|------|
| baseline (Qwen3-4B) | 50.4% | 无 LoRA | run 0057 |
| Qwen3-1.7B non-think | 34.5% | `OPENCLAW_MODEL_REASONING=0` | run 0117 |
| Qwen3-1.7B think | **53.0%** | `OPENCLAW_MODEL_REASONING=1` | run 0118 |
| v1 | **66.0%** | terminal-only, BATCH=2, 8 steps | step8 最佳，step17 退化到 0% (崩溃) |
| v2 | 46.3% | turn-level PRM, oracle-judge, BATCH=4, KL | rubric 质量差导致退化 |
| v3 step16 | 49.7% | turn-level PRM, oracle-judge, rubric fix, BATCH=2, KL | 没超过 v1 |
| v3 step10 | 7.5% | 同上 | checkpoint 损坏，输出为空 |
| v4.2 step8 | 52.6% | turn-level PRM, self-judge, BATCH=2, 无 KL, 8 steps | run 0088；几乎无提升 |
| v4.3 | 训练中（step9/12） | terminal-only, BATCH=2, 12 steps, KL=0.01, -1 惩罚, whiten ε=1.0 | checkpoint: reinforce_lora_v4.3 |
| v4.4 | 待启动 | self-judge PRM, 其余同 v4.3 | v4.3 的 PRM 对照组 |

---

## ⚠️ DPO Training Code Bug (Critical!)

**发现时间**: 2026-04-21  
**文件**: `rl/train/train_dpo_lora.py` (已删除)  
**影响**: 所有 DPO 训练（v2, focused, mixed）都受影响

### Bug 1: tool_calls 被序列化成文本 (Line 36-38)
```python
if m.get("tool_calls"):
    tc_text = json.dumps(m["tool_calls"], ensure_ascii=False)
    content = (content + "\n" + tc_text).strip()
```
**问题**: 把 tool_calls 对象转成 JSON 字符串附加到 content，模型学不到正确的 tool_calls 格式。

**后果**: 模型学到的是在 content 里输出 `[{"function": {"name": "exec", ...}}]` 文本，而不是真正的 tool_calls 结构。

### Bug 2: tool role 被转换成 user (Line 41-44)
```python
elif role == "tool":
    clean.append({"role": "user", "content": f"[tool result] {content}"})
```
**问题**: 完全破坏了 tool 的语义，模型学到的是 user 说 "[tool result] ..."。

**后果**: 模型不知道 tool results 的真实格式，可能学会输出假的 tool results。

### Bug 3: 字符串切片不可靠 (Line 71)
```python
completion_text = full_text[len(prompt_text):]
```

### Bug 4: TRL 数据格式错误 - prompt + chosen 重复 (2026-04-22)

**问题**: TRL 会做 `prompt + chosen` 的 list concatenation，但我们的数据中 `chosen` 包含了完整对话（包括第一个 user message），导致第一个 user message 重复。

**表现**: 
- 所有 16 个样本都报 "Mismatch between tokenized prompt and the start of tokenized prompt+chosen" 警告
- TRL 检查 `tokenize(prompt)` 是否匹配 `tokenize(prompt + chosen)` 的开头，但因为重复导致不匹配

**错误代码** (`train_dpo_lora_fixed.py` 初版):
```python
prompt_msgs = [chosen_msgs[0]]  # 第一个 user message
chosens.append(chosen_msgs)     # 完整的 3 个 messages (包括第一个 user message)

# TRL 内部会做: prompt + chosen
# 结果: [user_msg, user_msg, assistant_msg, tool_msg]  # user_msg 重复了！
```

**正确代码** (`train_dpo_lora_fixed.py` 修复后):
```python
prompt_msgs = [chosen_msgs[0]]       # 第一个 user message
chosen_completion = chosen_msgs[1:]  # 跳过第一个 user message，只要 completion
chosens.append(chosen_completion)

# TRL 内部会做: prompt + chosen
# 结果: [user_msg, assistant_msg, tool_msg]  # 正确！
```

**验证方法**:
```python
# 检查 TRL 的 tokenization 逻辑
prompt_tokens = tokenizer.apply_chat_template(prompt_msgs, tokenize=True)
chosen_tokens = tokenizer.apply_chat_template(prompt_msgs + chosen_completion, tokenize=True)

# 应该匹配
assert chosen_tokens[:len(prompt_tokens)] == prompt_tokens
```

**教训**:
1. TRL 的 DPO 数据格式：`prompt` 和 `chosen/rejected` 是分开的，TRL 会自动 concatenate
2. 不要假设数据格式，要看 TRL 源码确认
3. Tokenizer mismatch 警告必须重视，说明数据格式有问题
4. 修复后训练完全没有警告，loss 正常收敛

**相关文件**:
- `rl/train/train_dpo_lora_fixed.py` - 修复后的训练脚本
- `rl/train/debug_trl_exact.py` - 调试 TRL tokenization 的脚本

---

## ⚠️ Qwen3 Tool Calls 支持 (重要澄清)

**误解**: Qwen3-1.7B 不支持 tool_calls

**真相**: 
- ✅ Qwen3-1.7B **原生支持** tool_calls
- ✅ Chat template 有完整的 tool_calls 处理逻辑
- ✅ 支持 `tools` 参数、`tool_calls` 字段、`tool` role

**验证**:
```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
messages = [
    {"role": "user", "content": "test"},
    {"role": "assistant", "content": "ok", "tool_calls": [...]},
    {"role": "tool", "content": "result"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
# 输出包含 <tool_call> 和 <tool_response> 标签
```

**DPO 训练正确做法**:
1. 数据保留原始 tool_calls 结构（list of dicts）
2. 使用 TRL 的 Messages 格式
3. 让 Qwen3 的 chat template 自动处理
4. **不要**手动序列化 tool_calls
5. **不要**转换 tool role

---

## ⚠️ DPO Training Code Bug (Critical!)
**问题**: 如果 chat template 有特殊格式（如 special tokens），切片位置可能错误。

**后果**: prompt/completion 边界错误，label masking 失效。

### 修复方案

**已创建**: `rl/train/train_dpo_lora_fixed.py`

**关键改动**:
1. ✅ 直接传递 messages 格式给 TRL，不手动序列化
2. ✅ 保持 tool_calls 和 tool role 原样
3. ✅ 让 TRL 的 DPOTrainer 处理 chat template 和 prompt/completion 切分

**数据格式**:
```python
{
    "prompt": [{"role": "user", "content": "..."}],  # 只有 user prompt
    "chosen": [...],   # 完整 messages (包含 tool_calls)
    "rejected": [...]  # 完整 messages
}
```

TRL 会自动:
- 应用 chat template（正确处理 tool_calls）
- 切分 prompt/completion（基于 tokenizer）
- Label masking（只对 assistant 部分计算 loss）

**下一步**: 用 fixed 版本重新训练所有 DPO checkpoints。

---

## 核心创新：Focused DPO

### 问题发现

**DPO v2 训练 loss 降到 0，但 task_18 完全失败（2.5%）**

分析发现：
- DPO 数据：chosen 10.3 msgs (4 exec) vs rejected 20.6 msgs (0-1 exec)
- 模型可能学到 "短轨迹 vs 长轨迹"，而不是 "exec vs read"
- Exec 信号被稀释在长轨迹中

### 解决方案：Focused DPO

**核心思想：截断到关键决策点，强化对比信号**

- 找到 chosen 中最后一个 exec+openpyxl 调用
- 截断 chosen 到该点之后（包含 exec 结果）
- 截断 rejected 到相同长度
- 结果：chosen 7.1 msgs vs rejected 7.1 msgs，**同样长度下对比 exec vs read**

**数据统计：**
- 19 个 focused pairs
- Chosen: 10.3 → 7.1 messages (31% 压缩)
- Rejected: 20.6 → 7.1 messages (66% 压缩)
- 对比更清晰：相同长度，不同工具选择

**训练配置：**
- Base: Qwen3-1.7B (非 DPO)
- LoRA rank: 128 (vs v2 的 32)
- Epochs: 5 (vs v2 的 3)
- Data: `dpo_pairs_focused_train.jsonl`
- Script: `rl/train/run_dpo_focused.sh`

**创新意义：**
- 传统 DPO：训练完整轨迹，信号稀释
- Focused DPO：只训练关键决策，信号集中
- 适用于所有需要学习特定工具/行为的场景

---

## 核心教训

### 1. rubric 质量 > 算法超参
v2 退化的根因是 rubric 描述了 OpenClaw 里不存在的工具（awk、exec retry），oracle judge 按错误 rubric 打分，模型学到了错误策略。错误的 rubric 比没有 PRM 更差。

### 2. oracle-judge 噪声大，self-judge 更稳
v1（self-judge）> v2/v3（oracle-judge）。oracle judge 对 agent 行为的评分方差高，rubric 覆盖不完整时直接污染梯度。self-judge 用训练中的模型自己打分，噪声更小，且无需外部 API。

### 3. val-core reward ≠ 真实任务分数
val-core reward 是 EMA-normalized advantage，不是 judge 的绝对分数。step10 的 val reward 最高（+0.0467）但实际 benchmark 只有 7.5%——checkpoint 已经损坏（输出为空）。**不能用 val-core reward 选 checkpoint，必须跑真实 benchmark。**

### 4. checkpoint 损坏的症状
transcript 里 `assistant content: []`，所有任务得 0 分。原因是模型过训练后输出 EOS token 或空序列。v3 step10 就是这个情况。

### 5. task_02 的 reward hacking
模型发现"拒绝 = terminal reward 0"比"写错文件 = 负 advantage"更安全，学会了直接拒绝而不调用任何工具。rubric 里写"拒绝是最差结果"对 self-judge 效果有限——根本解法是让拒绝的 terminal reward 显式为 -1（v5 考虑）。

### 6. task_24 的 spurious correlation
模型在构造搜索 query 时自动加 "2023"，是训练数据里的 pattern。rubric 里写具体年份（2023/FTX）会 overfit，改成通用描述"不能用训练记忆里的过期数据"。

---

## 训练流程

### RunPod 信息
- SSH: `ssh root@216.81.248.115 -p 15416 -i ~/.ssh/id_ed25519`
- tmux session: `rl_train`
- 训练日志: `/tmp/train_v4.log`
- checkpoint 路径: `/workspace/pinchbench-skill/rl/checkpoints/reinforce_lora_v4/`
- oracle judge API key: `~/.pinchbench_env`（Mac 本地）

### 启动训练
```bash
# 1. 清显存
pkill -f vllm; sleep 3; nvidia-smi

# 2. 启动（在 tmux 里）
cd /workspace/pinchbench-skill && source ~/.pinchbench_env
OUTPUT_DIR=rl/checkpoints/reinforce_lora_v4 BATCH_SIZE=2 TOTAL_TRAINING_STEPS=8 \
  bash rl/train/run_reinforce_lora.sh 2>&1 | tee /tmp/train_v4.log
```

### 同步代码到 RunPod
```bash
rsync -av -e "ssh -i ~/.ssh/id_ed25519 -p 15416" \
  rl/agent_loop/reward.py \
  rl/train/run_reinforce_lora.sh \
  root@216.81.248.115:/workspace/pinchbench-skill/rl/agent_loop/
```
注意：rsync 不能在同一命令里混 local 和 remote 目标，要分目录分次传。

### 训练坑
- **TASK_EMA_INIT=0.3 太高**：所有 advantage 为负，模型无法学习。改为 0.1。
- **MAX_TURNS=10 不够**：task_16 需要读 13 封邮件，10 turns 不够。改为 16。
- **BATCH_SIZE=8 OOM**：A100 80G 上跑不起来，用 BATCH_SIZE=2。
- **LoRA rank mismatch**：vLLM 默认 `max_lora_rank=16`，但训练用 rank=32。启动 vLLM 时必须加 `--max-lora-rank 64`。
- **pkill -f vllm 会断 SSH**：用 tmux send-keys 发 C-c 再 pkill，或者直接在 tmux 窗口里操作。
- **OUTPUT_DIR 环境变量无效**：训练脚本用 `RUN_VERSION` 拼路径（`reinforce_lora_v4.3`），直接传 `OUTPUT_DIR=...` 会被忽略，checkpoint 存到默认的 `reinforce_lora/`。启动训练必须用 `RUN_VERSION=v4.x`，否则下一次训练会覆盖上一次的 checkpoint。
- **advantage 极端值（-50 ~ -100）**：根因是 veRL 的 `masked_whiten` 用 `epsilon=1e-8`，reward 稀疏时（只有 `<|im_end|>` token 有值）var 趋近于 0，`rsqrt(var + 1e-8)` 爆炸。修法：直接改 veRL 源码 `/usr/local/lib/python3.12/dist-packages/verl/utils/torch_functional.py` 第 336 行，把 `1e-8` 改为 `1.0`。改后 advantage 范围从 ±100 收敛到 ±0.2。
  ```python
  # 改前
  whitened = (values - mean) * torch.rsqrt(var + 1e-8)
  # 改后
  whitened = (values - mean) * torch.rsqrt(var + 1.0)
  ```
  注意：每次 RunPod 重建环境后需要重新 patch。`norm_adv_by_std_in_grpo=False` 无效，因为问题在 `masked_whiten` 本身，所有 estimator（reinforce_plus_plus / grpo / rloo）都调用了它。
- **Qwen3.5-2B + veRL 0.7.1 / vLLM 0.19 训练启动有兼容坑**：RunPod 上 `transformers 4.57.6` 不认识 `model_type=qwen3_5`，`Qwen/Qwen3.5-2B` 会在 `actor_rollout_init_model()` 报 `model type qwen3_5` not recognized，训练连第一批 rollout 都进不去。不要直接升到 `transformers 5.x`：实测会和 `vllm 0.19.0` / `compressed-tensors` 的 `transformers<5` 依赖冲突。也不要简单注册 `qwen3_5 -> Qwen3Config`：Qwen3.5 的顶层 config 是 wrapper，真正 LM 参数在 `text_config`；错误 alias 会导致 `vocab_size=151936`、`pad_token_id=248044`，然后报 `AssertionError: Padding_idx must be within num_embeddings`。当前仓库用本地 patch 解决：
  - `sitecustomize.py`：Ray worker 继承 `PYTHONPATH` 后自动加载 Qwen3.5 兼容 shim。
  - `rl/transformers_qwen3_5_patch.py`：仅当 checkpoint 顶层 `model_type == qwen3_5` 且存在 `text_config` 时，把 `text_config` 转成 `Qwen3NextConfig / Qwen3NextForCausalLM`；普通 `Qwen3` 不走这个分支。
  - `rl/train/launch_main_ppo.py`：先 import 本地兼容 patch，再 `runpy.run_module("verl.trainer.main_ppo")`。
  - `rl/train/run_reinforce_lora.sh`：入口从 `python3 -m verl.trainer.main_ppo` 改为 `python3 rl/train/launch_main_ppo.py`。
  启动前可确认：
  ```bash
  cd /workspace/pinchbench-skill
  PYTHONPATH=/workspace/pinchbench-skill python3 - <<'PY'
  import sitecustomize
  from transformers import AutoConfig
  for model in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-2B"]:
      cfg = AutoConfig.from_pretrained(model)
      print(model, type(cfg).__name__, cfg.model_type, getattr(cfg, "vocab_size", None), getattr(cfg, "hidden_size", None))
  PY
  ```
  预期：`Qwen/Qwen3-0.6B` 仍是 `Qwen3Config qwen3`；`Qwen/Qwen3.5-2B` 是 `Qwen3NextConfig qwen3_next`。
- **Qwen3.5-2B 没有 `generation_config.json`，veRL / Transformers fallback 也会炸**：修完 `qwen3_5` 后，veRL 会继续在 `verl.utils.model.get_generation_config()` 里找 `generation_config.json`；找不到后走 `GenerationConfig.from_model_config(config)`，可能报 `AttributeError: 'dict' object has no attribute 'to_dict'`。即使 veRL 的 `get_generation_config()` 被兜住，Transformers 模型构造阶段也可能自己再调用 `GenerationConfig.from_model_config(config)`，因此当前仓库用 `rl/verl_qwen3_5_generation_patch.py` 同时 patch 两处：对 `Qwen3.5` 的 veRL `get_generation_config()` 直接返回默认 `GenerationConfig()`，并对 `GenerationConfig.from_model_config()` 的 Qwen3.5 `dict.to_dict()` 崩溃做默认兜底。启动前可验证：
  ```bash
  cd /workspace/pinchbench-skill
  PYTHONPATH=/workspace/pinchbench-skill python3 - <<'PY'
  import rl.transformers_qwen3_5_patch
  import rl.verl_qwen3_5_generation_patch
  from verl.utils.model import get_generation_config
  print(type(get_generation_config("Qwen/Qwen3.5-2B")).__name__)
  PY
  ```
  正常输出应包含 `GenerationConfig`。注意：这个 patch 必须通过 `sitecustomize.py` 自动加载，因为 Ray worker 不一定经过 `rl/train/launch_main_ppo.py` 的主进程 import；只挂 launcher 会导致 worker 里仍调用原始 `verl.utils.model.get_generation_config()`。这两个问题都发生在模型初始化阶段，跟 rollout、reward、advantage 无关；只有过了这里，才开始检查 no-think 轨迹和 advantage 分布。
- **Qwen3.5-2B / Qwen3Next 初始化会先爆 CPU RAM，不是 GPU OOM**：实测 Qwen3.5-2B 走 `Qwen3NextConfig` 后可越过 `Padding_idx`，但 Ray worker 在 `actor_rollout_init_model()` 期间占用约 41.6GB host RAM，节点 cgroup 只有约 `46.57GB`，GPU 显存仍只有 `266MiB`。尝试结果：
  - `RAY_NUM_CPUS=4` + Ray 默认 95%：在 `44.39GB / 46.57GB` 被 Ray 杀。
  - `RAY_NUM_CPUS=4 RAY_memory_usage_threshold=0.99`：在 `46.11GB / 46.57GB` 被 Ray 杀。
  - `RAY_NUM_CPUS=3`：资源不足，TaskRunner 占 1 CPU 后，Worker placement group 还要 `CPU:3 + GPU:1`，pending。
  - `RAY_NUM_CPUS=4 RAY_memory_usage_threshold=0.999`：在 `46.56GB / 46.57GB` 被 Ray 杀。
  - `RAY_memory_monitor_refresh_ms=0`：Ray 不杀，但 pod/sshd 开始 reset，说明不是 Ray 阈值误杀，而是 cgroup 内存真的贴顶。
  结论：当前这类 48G GPU 但 host RAM/cgroup 约 46GB 的 pod 不适合用当前 veRL+Qwen3Next 训练 Qwen3.5-2B。需要换 host RAM 更大的 pod，或换能原生/低内存加载 Qwen3.5 的栈；否则训练还没进 rollout 就卡在模型初始化。`rl/train/run_reinforce_lora.sh` 已提供 offload 开关：`ACTOR_PARAM_OFFLOAD`、`ACTOR_OPTIMIZER_OFFLOAD`、`REF_PARAM_OFFLOAD`，默认保持历史 `True`；但本次瓶颈在初始化期 host RAM 峰值，单纯调 Ray 阈值不够。
- **flash-linear-attention / causal-conv1d 不要源码编译**：Qwen3Next 日志会提示缺快路径并 fallback 到 torch implementation。不要在 pod 上 `pip install` 触发本地编译；只能装与当前 Python/Torch/CUDA 精确匹配的预编译 wheel。没有 wheel 就先接受 torch fallback。
- **Qwen3.5-2B 的 vLLM LoRA 同步会在 `update_weights()` 崩 `StopIteration`**：这次不是 rollout 或 OpenClaw 的问题，而是 vLLM 在 `TensorLoRARequest` 同步阶段，`LoRAModelManager._create_merged_loras_inplace()` 没有收到任何匹配的 LoRA layer，`next(iter(lora_model.loras.values()))` 直接 `StopIteration`。症状是：
  - vLLM 日志先出现大量 `Qwen3_5ForConditionalGeneration` / `visual.* ... ignored` warning。
  - 随后在 `collective_rpc method failed`，根因是 worker 里的 `add_lora()` 走到空 `lora_model.loras`。
  - 训练主进程最终在 `checkpoint_manager.update_weights()` 处退出。
  这说明 Qwen3.5 的模型/LoRA 兼容 patch 还没有完全把 rollout 侧收敛到 text-only，当前不建议继续在这条链路上硬耗时间。
- **Qwen3-1.7B / LoRA 空集合最终解法**：不是把空 LoRA 静默跳过。最初 vLLM 在 `update_weights()` 里崩 `StopIteration`，堆栈是 `trainer.fit() -> checkpoint_manager.update_weights() -> vLLMHttpServer.collective_rpc() -> add_lora() -> LoRAModelManager._create_merged_loras_inplace()`，最后 `next(iter(lora_model.loras.values()))` 发现 `lora_model.loras` 为空。临时加过 `rl/verl_vllm_lora_empty_guard_patch.py` 做诊断和 hard fail，但真正修复是禁用 `layered_summon`。
- **根因：`layered_summon=True` 收不到 Qwen3-1.7B 的 LoRA tensor**。`layered_summon=True` 会让 veRL 走 `layered_summon_lora_params()`，该函数靠固定 prefix 扫 FSDP 子模块；Qwen3-1.7B 当前 FSDP/PEFT 命名不匹配，导致 `collect_lora_params()` 收集到 `0` 个 LoRA tensor。vLLM 侧日志表现为：
  ```text
  [vllm_lora_debug] vLLM _update_weights count=0 base_sync_done=True has_peft_config=True sample=[]
  empty LoRA model reached vLLM merge path
  StopIteration
  ```
  正确启动方式：设置 `ROLLOUT_LAYERED_SUMMON=False`，让普通 `FSDP.summon_full_params + get_peft_model_state_dict()` 收集 LoRA。健康信号：
  ```text
  collect_lora_params done count=392
  [pinchbench_lora_only_ckpt] saved 392 LoRA tensors
  ```
  `count=0` 必须视为训练无效，不能继续跑。
- **LoRA-only checkpoint 保存也要跟随 `layered_summon` 配置**：之前训练同步用 `ROLLOUT_LAYERED_SUMMON=False` 已经能收集到 LoRA，但保存 patch 还固定用 layered summon，导致写出 16 bytes 的空 `adapter_model.safetensors`。`rl/verl_lora_only_ckpt_patch.py` 已改成读取 rollout 的 `layered_summon` 配置，和训练同步路径一致。正常 adapter 大小约 `139512944 bytes`。
- **Qwen3.5 训练链路的结论**：在这个 pod / veRL / vLLM 组合下，Qwen3.5-2B 的训练初始化虽然能过，但 rollout 侧 LoRA 同步不稳定，最终在 vLLM `update_weights` 阶段崩掉；工程成本过高，性价比不如直接换回已知可跑的更小模型做 RL8 验证。

---

## Benchmark 流程

### 完整步骤
```bash
# 1. RunPod 上启动 vLLM（注意 --max-lora-rank 和 --lora-modules alias）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules Qwen3-4B=/workspace/pinchbench-skill/rl/checkpoints/reinforce_lora_v4.2/global_step_8/actor/lora_adapter \
  --port 8021 \
  --max-model-len 40960 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  2>&1 | tee /tmp/vllm.log
# ⚠️ --enable-auto-tool-choice 必须加！
# tool parser 必须匹配模型：
#   Qwen3/Qwen3-4B:     --tool-call-parser hermes
#   Qwen3.5 系列:       --tool-call-parser qwen3_xml
# parser 选错会导致 vLLM 返回 tool_calls=[] 或解析报错，OpenClaw 只停在 toolUse，
# 轨迹没有 tool result，RL8 会从正常分数掉到个位数。

# 2. Mac 上建隧道（用 autossh 保持稳定）
lsof -ti:18021 | xargs kill -9 2>/dev/null
autossh -M 0 -fN \
  -L 127.0.0.1:18021:127.0.0.1:8021 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  root@216.81.248.115 -p 15416 -i ~/.ssh/id_ed25519

# 3. 验证
curl http://127.0.0.1:18021/v1/models  # 应该列出 Qwen3-4B

# 4. 跑 benchmark
MODEL=Qwen3-4B BASE_URL=http://127.0.0.1:18021/v1 bash scripts/run_bench_rl8.sh
```

### Benchmark 坑
- **`--enable-auto-tool-choice` 必须加，`--tool-call-parser` 必须匹配模型**：缺少 auto tool choice 时，OpenClaw 发带 tools 的 chat/completions 请求会得到 400 Bad Request，所有任务空输出，全 0 分。parser 选错同样会让工具链断掉。
- **Qwen3.5 系列必须用 `--tool-call-parser qwen3_xml`，不能用 `hermes`**：Qwen3.5 实际输出 `<tool_call><function=...><parameter=...>` XML 风格。用 `hermes` 时 vLLM 会按 JSON 解析，raw response 里 `tool_calls: []`，日志可能出现 `hermes_tool_parser.py JSONDecodeError`。症状是 transcript 只有 6 行、assistant `stopReason: toolUse`、没有 `toolCall` / tool result，2B 和 4B 得分都会异常接近个位数。实测 Qwen3.5-4B 从 `hermes` 的 RL8 7.5% 修正到 `qwen3_xml` 后为 79.5%（run 0111）。
- **Qwen3.5-2B 启动时不要额外挂 `--reasoning-parser deepseek_r1`**：实测 `qwen3_xml + deepseek_r1` 会让 2B 的 XML tool call 进入 response `reasoning`，最终 `tool_calls: []`；去掉 `--reasoning-parser deepseek_r1` 后 raw response 正常返回结构化 `tool_calls`，RL8 transcript 也有成对 `toolCall/toolResult`。这不是证明 2B 没有 reasoning 能力，只是当前 vLLM parser 组合对 2B 不安全。
- **benchmark 脚本在 Mac 本地跑**，不是在 RunPod 上。OpenClaw agent 在 ECS（8.163.82.224）。
- **vLLM lora-modules alias 必须和 MODEL 参数一致**：`--lora-modules Qwen3-4B=...` 对应 `MODEL=Qwen3-4B`。用不同 alias 会导致 404 → 空输出 → 全 0 分。
- **隧道端口**：benchmark 用 18021→8021，不要用 18010（那是 base model 的端口）。
- **隧道断了 curl 返回 exit code 7**（connection refused）或 exit code 56（network error）。重建隧道即可。
- **vLLM 没完全 ready 时 benchmark 会得空输出**：等到 `curl /v1/models` 返回正常再跑。
- **Qwen3 thinking mode**：模型默认开启 `<think>` 块，OpenClaw 设置 `thinkingLevel: off` 来关闭。如果 thinking 没关掉，输出全是 `<think>` 内容，assistant content 为空。

### 结果文件
- benchmark 结果：`results/{run_id}_qwen3-4b.json`
- transcript：`results/{run_id}_transcripts/`

---

## SFT 踩坑：tool_calls loss masking

### 问题
task_18_spreadsheet_summary 需要 exec 工具解析 xlsx 文件（用 openpyxl），但 DPO 模型从不调 exec，只用 read 读 xlsx 得到二进制乱码。RL 训练也无法激活 exec（step 4/8 都是 2.5%）。

### 尝试 1：全轨迹 SFT（失败）
- 数据：20 个完整 teacher 轨迹（9-13 turns）
- 结果：benchmark 19.2%，但 transcript 分析显示**从未调用 exec**，只用 read/write
- 根因：exec 调用只在 turn 2-3，占总 token 的 ~20%，信号被稀释

### 尝试 2：DataCollatorForLanguageModeling（错误）
- 问题：对**所有 token** 计算 loss，包括 user message、tool result
- 应该只对 assistant 生成的部分（content + tool_calls）计算 loss

### 解决方案：手动 label masking
在 Qwen chat template 里，tool_calls 被序列化为：
```
<|im_start|>assistant
<tool_call>
{"name": "exec", "arguments": "{...}"}
</tool_call><|im_end|>
```

需要：
1. 找到所有 `<|im_start|>assistant\n` 位置
2. 从该位置到下一个 `<|im_start|>` 之间的 token，labels = input_ids（计算 loss）
3. 其余 token labels = -100（mask 掉）

这样 tool_calls 的 JSON 内容会被正确训练。

### 最终方案：focused SFT（失败）
- 数据：只保留 user prompt + 前几轮（到第一个 exec+openpyxl 调用 + 其结果）
- 压缩：10.3 turns → 3.9 turns，38.5% token
- 训练：2 epochs，lr=5e-6，手动 label masking
- 代码：`rl/train/train_sft_lora.py`，`rl/data/generated/.../sft_exec_focused_train.jsonl`
- **结果：3 次 benchmark 都是 0 分，完全失败**

### vLLM benchmark 踩坑
1. **Model name 不匹配**：vLLM 默认用 checkpoint 路径作为 model name，需要 `--served-model-name Qwen3-1.7B` 匹配 benchmark 请求
2. **Tool calls 不支持**：需要 `--enable-auto-tool-choice --tool-call-parser hermes` 才能处理 tool_calls
3. **Context length 太小**：默认 8192，task_18 prompt 有 55k+ chars，需要 `--max-model-len 32768`
4. **GPU OOM 循环**：`pkill -9 -f python` 不彻底，老进程一直占显存（41 GiB），需要手动 `nvidia-smi --query-compute-apps=pid | xargs kill -9`

### SFT 失败根因分析

**Bug 1: Label masking 完全失效**
- 原实现用 token-by-token 匹配 `<|im_start|>assistant`，但这是多个 token，匹配失败
- 结果：所有 assistant tokens 都被 mask（0.0% unmasked），exec 完全没有 loss
- 修复：用 `return_offsets_mapping` + regex 匹配 text，正确 unmask assistant 部分（13.2% unmasked）

**Bug 2: 数据提取不完整**
- 原提取逻辑：找到第一个 exec+openpyxl 调用，截断到下一个 message
- 问题：如果 assistant 有多个 tool_calls（如 `['read', 'exec']`），只包含了第一个 tool result
- 结果：exec 的真正结果（openpyxl 输出）被截断，模型看不到 exec 的效果
- 修复：找到**最后一个** exec+openpyxl（最完整的用法），包含所有 tool results
- 数据量：3.9 turns → 7.1 turns

**Bug 3: RL FIFO compaction 删除关键 turns**
- 当前逻辑：FIFO，从头部删除最老的 turns
- 问题：exec 通常在 turn 2-4，如果后续有 10+ turns，exec turn 会被删掉
- 结果：训练时看不到 exec，梯度无法优化 exec 行为
- 修复：改成 LIFO，从尾部删除最新的 turns，保护早期的关键 tool calls
- 代码：`rl/agent_loop/openclaw_agent_loop.py:238-276`

**SFT v2 结果（修复 Bug 1+2）**
- 训练：19 samples，7.1 turns avg，loss 1.042→1.082 (avg 1.11)
- Benchmark：3 次平均 11.1% (2.5%, 28.3%, 2.5%)
- **仍然没有调用 exec**，只用 read/write
- 根因：LoRA rank=32 太小 (1.98% params) + DPO base model 的 read bias 太强 + 样本太少（19个）

**结论**：SFT 方法失败。即使修复所有 bugs，小 LoRA 无法覆盖 DPO base model 的强 prior。建议：
1. 从 Qwen3-1.7B base（非 DPO）直接 SFT exec
2. 增加 LoRA rank 到 128 或 full fine-tuning
3. 增加样本到 100+ 和 epochs 到 10+

---

## OpenClaw Catalog Model Registration

### 请求链路
```
Mac benchmark script
  → bench-qwen3-4b agent (动态创建，配置写入 ECS)
    → model id: "Qwen3-4B"
    → baseUrl: http://127.0.0.1:18021/v1
      → SSH tunnel: 127.0.0.1:18021 → RunPod:8021
        → vLLM serving LoRA adapter
```

benchmark 脚本（`scripts/benchmark.py`）会自动在 ECS 上创建/更新 `bench-qwen3-4b` agent，不需要手动配置 models.json。只需保证 MODEL 和 vLLM lora-modules alias 一致即可。

---

## v4.2 配置与分析

| 参数 | 值 |
|------|----|
| REWARD_MODE | self-judge |
| BATCH_SIZE | 2 |
| TOTAL_TRAINING_STEPS | 8 |
| MAX_TURNS | 16 |
| KL loss | 关闭 |
| TASK_EMA_INIT | 0.1 |
| PRM ceiling | +0.2 |
| TERMINAL_REWARD_WEIGHT | 0.7 |
| terminal reward | {0, +1} |
| checkpoint | reinforce_lora_v4.2/global_step_8 |
| benchmark run | 0088，均分 52.6% |

### v4.2 失败根因
- **task_02（0分）**：模型调了 web_search + 2x web_fetch 找到数据，然后用文字说"已保存"，从未调 write 工具。terminal=0 对幻觉完成无惩罚。
- **task_12（0分）**：模型用 `read: config/*`（glob 语法），read 工具不支持 glob → ENOENT → 认为目录不存在 → 8 turns 放弃。是工具使用错误，非 RL 能高效修复的问题。
- **整体**：PRM self-judge 信号噪声大，对"找到信息但不写文件"惩罚不足，8 steps 基本没动基础分布。

### v4.2 rubric 变更（相对 v3）
- `task_02_stock`：去掉引用具体拒绝话术的条目，改为通用"文件必须存在"原则
- `task_24_polymarket_briefing`：去掉 2023/FTX 等具体年份，改为通用"不能用训练记忆里的过期数据"

---

## v4.3 配置

| 参数 | 值 |
|------|----|
| REWARD_MODE | **baseline**（纯 terminal，回到 v1 方式） |
| BATCH_SIZE | 2 |
| TOTAL_TRAINING_STEPS | 12 |
| MAX_TURNS | 16 |
| KL loss | 开启，coef=0.01 |
| TASK_EMA_INIT | 0.1 |
| TERMINAL_REWARD_WEIGHT | 1.0 |
| terminal reward | {-1, 0, +1}（新增 -1） |
| masked_whiten epsilon | 1.0（patched） |
| checkpoint | reinforce_lora_v4.3 |
| 训练日志 | RunPod /tmp/train_v4.3.log，tmux: rl_v43 |

### v4.3 核心改动
- **去掉 PRM**：回到 v1 的纯 terminal reward，消除 self-judge 噪声
- **-1 terminal 惩罚**：task_02_stock（stock_report.txt）、task_18_spreadsheet_summary（data_summary.md）目标文件不存在 → terminal = -1，直接惩罚幻觉完成行为
- **12 steps**：v1 step12 是历史最高（63.8%），8 steps 不够收敛
- **KL=0.01**：v4 无 KL 在 step17 崩溃，轻 KL 防止过训练
- **masked_whiten epsilon 1e-8→1.0**：修复 advantage 极端值问题

### 训练观测（step 1-9）
- advantage 范围 [-0.7, +8.3]，正常
- grad_norm 全程 0.006~0.074，稳定
- step3/6/7 出现成功 episode，学习信号有效

### 启动命令
```bash
RUN_VERSION=v4.3 BATCH_SIZE=2 TOTAL_TRAINING_STEPS=12 \
  REWARD_MODE=baseline PINCHBENCH_TERMINAL_REWARD_WEIGHT=1.0 \
  bash rl/train/run_reinforce_lora.sh 2>&1 | tee /tmp/train_v4.3.log
```

---

## v4.4 配置（待启动，v4.3 结束后）

v4.3 的对照组，唯一变量：加回 PRM（self-judge），验证在 -1 惩罚 + 稳定 advantage 的前提下 PRM 是否有帮助。

| 参数 | 值 |
|------|----|
v4.3 的对照组，加回 PRM（self-judge）+ 步数加长到 24，同时修复训练环境的 Brave search 配置（v4.3 训练时 ECS 没有 Brave key，task_02 rollout 全是 DDG 广告，reward=0）。

| 参数 | 值 |
|------|----|
| REWARD_MODE | **self-judge**（PRM） |
| BATCH_SIZE | 2 |
| TOTAL_TRAINING_STEPS | **24** |
| MAX_TURNS | 16 |
| KL loss | 开启，coef=0.01 |
| TASK_EMA_INIT | 0.1 |
| TERMINAL_REWARD_WEIGHT | 1.0 |
| terminal reward | {-1, 0, +1} |
| masked_whiten epsilon | 1.0（patched，每次重建环境需重新 patch） |
| BRAVE_API_KEY | 已写入 ECS `~/.pinchbench_env` + RunPod `~/.pinchbench_env` |
| checkpoint | reinforce_lora_v4.4 |
| 训练日志 | RunPod /tmp/train_v4.4.log，tmux: rl_v44 |
| 启动时间 | 2026-04-17 |

### 与 v4.3 的关键差异
- steps 12 → 24（更多训练）
- 加回 PRM self-judge
- ECS Brave search 修复（v4.3 训练时 task_02 一直得 0 reward）

### 启动命令
```bash
RUN_VERSION=v4.4 BATCH_SIZE=2 TOTAL_TRAINING_STEPS=24 \
  REWARD_MODE=self-judge PINCHBENCH_TERMINAL_REWARD_WEIGHT=1.0 \
  bash rl/train/run_reinforce_lora.sh 2>&1 | tee /tmp/train_v4.4.log
```

---

## v5 方向（待讨论）

- `task_24 prompt 注入日期`：在 task 定义里加 `Today's date is {date}`，需同步改 benchmark task 文件
- `n_votes=2 ensemble judge`：每个 turn 调用 judge 两次取均值，降低 PRM 方差
- SkillHub search：给 agent 加 clawdhub_search 工具，task_24（polymarket）等实时数据任务可通过安装专用 skill 解决，ROI 可能高于继续 RL
