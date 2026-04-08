#!/usr/bin/env bash
# veRL single-sample PG 训练脚本（RunPod 上执行）
#
# 用法：
#   # Step 1：准备数据
#   python rl/train/prepare_data.py \
#       --input rl/data/samples_rescored.jsonl \
#       --output-dir rl/data/verl/
#
#   # Step 2：跑训练
#   bash rl/train/run_verl.sh
#
# 算法：GPG（single-sample Policy Gradient）
#   - adv_estimator=gpg：单样本 PG，不需要组内对比，适合 Live-User 场景
#   - use_kl_loss=True：reference-policy KL 正则，防止策略跑偏
#   - LoRA 微调：节省显存，L4（24GB）可跑
#
# 依赖：
#   pip install verl vllm transformers peft

set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/rl/data/verl"
OUTPUT_DIR="${REPO_ROOT}/rl/checkpoints/verl"
REWARD_FN_PATH="${REPO_ROOT}/rl/train/reward_fn.py"

TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/val.parquet"

# ── 模型配置 ──────────────────────────────────────────────
MODEL="${VERL_MODEL:-Qwen/Qwen3-1.7B}"
N_GPUS="${VERL_N_GPUS:-1}"      # L4 单卡即可

echo "=============================="
echo "  veRL PinchBench RL 训练"
echo "  模型: ${MODEL}"
echo "  GPU 数: ${N_GPUS}"
echo "  数据: ${DATA_DIR}"
echo "  输出: ${OUTPUT_DIR}"
echo "=============================="

# 检查数据文件
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "错误：训练数据不存在: ${TRAIN_FILE}"
    echo "请先运行: python rl/train/prepare_data.py --input rl/data/samples_rescored.jsonl --output-dir rl/data/verl/"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gpg \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=pinchbench_rl \
    trainer.experiment_name="qwen3_1.7b_gpg_$(date +%Y%m%d_%H%M)" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="${OUTPUT_DIR}"
