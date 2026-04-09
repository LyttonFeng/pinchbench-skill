#!/usr/bin/env bash
# Online RL: REINFORCE++ + LoRA + OpenClaw Agent Loop
#
# Live-user 场景 (rollout.n=1)：每个 training step 从 8 个 task 采样 batch_size 个，
# 用当前 LoRA 策略跑 openclaw episode，拿 process + terminal reward，
# 做一次 REINFORCE++ update (token-level running mean baseline)。
#
# 三组 ablation 实验：
#   REWARD_MODE=baseline    → Mode A: 纯 terminal reward
#   REWARD_MODE=rule        → Mode B: 通用行为规则 + terminal (无 LLM 调用)
#   REWARD_MODE=self-judge  → Mode C: Qwen3-4B self-judge with rubric + 天眼 (默认)
#   REWARD_MODE=oracle-judge→ Mode D: qwen-plus judge (fallback)
#
# 用法：
#   # Step 1: 准备 prompt 数据
#   python rl/train/prepare_prompts.py --tasks-dir tasks/ --output-dir rl/data/prompts/
#
#   # Step 2: 设置环境变量
#   export OPENCLAW_HOST=8.163.82.224       # 阿里云 ECS
#   export OPENCLAW_USER=root
#   export JUDGE_API_KEY=sk-xxx             # DashScope API key
#   export REWARD_MODE=oracle               # ablation mode
#
#   # Step 3: 启动训练
#   bash rl/train/run_reinforce_lora.sh
#
# 依赖：pip install verl vllm transformers peft aiohttp

set -euo pipefail

# ── 路径配置 ──
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/rl/data/prompts"
OUTPUT_DIR="${REPO_ROOT}/rl/checkpoints/reinforce_lora"
AGENT_LOOP_CONFIG="${REPO_ROOT}/rl/agent_loop/config.yaml"
REWARD_MANAGER_PATH="${REPO_ROOT}/rl/train/reward_manager.py"

TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/val.parquet"

# ── 模型配置 ──
MODEL="${VERL_MODEL:-Qwen/Qwen3-4B}"
N_GPUS="${VERL_N_GPUS:-1}"

# ── Python path (确保 rl 包可以被 import) ──
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# 使用 SDPA 代替 FlashAttention2（避免 flash_attn 包兼容性问题）
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# ── 训练超参 ──
BATCH_SIZE="${BATCH_SIZE:-8}"          # 8 个 task 各 1 条
MICRO_BATCH="${MICRO_BATCH:-4}"        # L40S 48GB VRAM
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LR="${LR:-2e-5}"
REWARD_MODE="${REWARD_MODE:-self-judge}"  # baseline / rule / self-judge / oracle-judge

# ── 环境变量检查 ──
echo "=============================="
echo "  veRL Online RL (REINFORCE++ + LoRA)"
echo "  模型: ${MODEL}"
echo "  LoRA rank: ${LORA_RANK}"
echo "  GPU 数: ${N_GPUS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Reward mode: ${REWARD_MODE}"
echo "  OpenClaw host: ${OPENCLAW_HOST:-localhost}"
echo "  Judge model: ${JUDGE_MODEL:-qwen-plus}"
echo "  数据: ${DATA_DIR}"
echo "  输出: ${OUTPUT_DIR}"
echo "=============================="

# 检查 prompt 数据
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "训练数据不存在: ${TRAIN_FILE}"
    echo "请先运行: python rl/train/prepare_prompts.py"
    exit 1
fi

# 设置环境变量供 agent loop 和 reward manager 使用
export PINCHBENCH_DIR="${REPO_ROOT}"
export REWARD_MODE="${REWARD_MODE}"
# PRM self-judge 走 RunPod 本地 vLLM（和 agent 共享同一个模型）
export PRM_VLLM_BASE_URL="${PRM_VLLM_BASE_URL:-http://localhost:8000/v1}"
export PRM_MODEL="${PRM_MODEL:-Qwen3-4B}"
export PRM_API_KEY="${PRM_API_KEY:-dummy}"

mkdir -p "${OUTPUT_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${BATCH_SIZE}" \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH}" \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30 \
    actor_rollout_ref.rollout.agent.default_agent_loop=openclaw_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="${REWARD_MANAGER_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name=pinchbench_rl \
    trainer.experiment_name="reinforce_lora_${REWARD_MODE}_$(date +%Y%m%d_%H%M)" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${OUTPUT_DIR}"
