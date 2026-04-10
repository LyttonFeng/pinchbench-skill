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
#   export DASHSCOPE_API_KEY=sk-xxx         # DashScope API key
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
# 仓库根 sitecustomize.py：默认可选 patch veRL（空 rollout_probs_diff、best ckpt 等），见 PINCHBENCH_DEBUG_METRICS_PATCH / PINCHBENCH_BEST_CKPT
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# 使用 SDPA 代替 FlashAttention2（避免 flash_attn 包兼容性问题）
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
# 注意: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 与 vLLM 0.19 的 CuMemAllocator 不兼容

# ── 训练超参 ──
BATCH_SIZE="${BATCH_SIZE:-1}"          # ECS 4核8G 资源有限，单并发最稳定
MICRO_BATCH="${MICRO_BATCH:-1}"        # 与 batch_size 匹配
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LR="${LR:-2e-5}"
REWARD_MODE="${REWARD_MODE:-self-judge}"  # baseline / rule / self-judge / oracle-judge
# vLLM rollout：OOM 时先降 VLLM_GPU_MEM_UTIL（如 0.22）或 VLLM_MAX_MODEL_LEN（如 16384）
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.28}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-18432}"

# Checkpoint 磁盘（RunPod /workspace 常不大；满盘会导致 torch.save zip 报错）
# 每次保存会写一个目录: ${OUTPUT_DIR}/global_step_{N}/
#   - actor/          FSDP 权重分片 + optimizer（体积最大，数 GB 级）
#   - actor/lora_adapter/  仅 LoRA（adapter_model.safetensors，几十 MB，推理主要用这个）
#   - data.pt         dataloader 状态（很小）
#   - latest_checkpointed_iteration.txt 在 OUTPUT_DIR 根目录
#
# PINCHBENCH_BEST_CKPT=1（默认）: 通过 repo 根目录 sitecustomize + rl/verl_best_ckpt_patch 在每次 val 后
# 按 val-core/*/reward/mean@* 只保留历史最佳 global_step_*；此时应让 save_freq 与 test_freq 一致，
# 否则 val 步上没有新 checkpoint，剪枝不会跑、旧目录会堆在盘上。
PINCHBENCH_BEST_CKPT="${PINCHBENCH_BEST_CKPT:-1}"
export PINCHBENCH_BEST_CKPT
TEST_FREQ="${TEST_FREQ:-5}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"    # BEST_CKPT=0 时：只保留最近 N 个 global_step_*
MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-1}"  # 无 critic 时无影响

if [ "${PINCHBENCH_BEST_CKPT}" = "1" ]; then
  SAVE_FREQ="${SAVE_FREQ:-${TEST_FREQ}}"
  HYDRA_MAX_ACTOR_KEEP='trainer.max_actor_ckpt_to_keep=null'
else
  SAVE_FREQ="${SAVE_FREQ:-20}"
  HYDRA_MAX_ACTOR_KEEP="trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}"
fi

if [ "${PINCHBENCH_BEST_CKPT}" = "1" ] && [ "${SAVE_FREQ}" != "${TEST_FREQ}" ]; then
  echo "WARN: PINCHBENCH_BEST_CKPT=1 建议 SAVE_FREQ==TEST_FREQ（当前 save_freq=${SAVE_FREQ} test_freq=${TEST_FREQ}），否则部分 val 步无新 checkpoint、无法按 val 清理旧目录。"
fi

# 与本次 trainer.experiment_name 同一时间戳；TensorBoard 单独子目录，便于打包给同事
RUN_STAMP="$(date +%Y%m%d_%H%M)"
EXPERIMENT_NAME="reinforce_lora_${REWARD_MODE}_${RUN_STAMP}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard/${EXPERIMENT_NAME}}"

# ── 环境变量检查 ──
echo "=============================="
echo "  veRL Online RL (REINFORCE++ + LoRA)"
echo "  模型: ${MODEL}"
echo "  LoRA rank: ${LORA_RANK}"
echo "  GPU 数: ${N_GPUS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Reward mode: ${REWARD_MODE}"
echo "  OpenClaw host: ${OPENCLAW_HOST:-localhost}"
echo "  OpenClaw remote activate: ${OPENCLAW_REMOTE_ACTIVATE_CMD:-<none>}"
echo "  Judge model: ${JUDGE_MODEL:-qwen-plus}"
echo "  Grading judge: ${PINCHBENCH_GRADE_JUDGE_MODEL:-qwen-plus} @ ${PINCHBENCH_GRADE_JUDGE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
echo "  数据: ${DATA_DIR}"
echo "  输出: ${OUTPUT_DIR}"
echo "  pinchbench_best_ckpt: ${PINCHBENCH_BEST_CKPT}  save_freq: ${SAVE_FREQ}  test_freq: ${TEST_FREQ}"
if [ "${PINCHBENCH_BEST_CKPT}" != "1" ]; then
  echo "  max_actor_ckpt: ${MAX_ACTOR_CKPT_TO_KEEP}"
fi
echo "  tensorboard: ${TENSORBOARD_DIR}  (需: pip install tensorboard)"
echo "  vLLM: gpu_memory_utilization=${VLLM_GPU_MEM_UTIL} max_model_len=${VLLM_MAX_MODEL_LEN}"
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
# ECS OpenClaw is expected to be directly on PATH after installer setup.
# Do not inject a remote activation command unless a site-specific setup requires it.
export OPENCLAW_REMOTE_ACTIVATE_CMD="${OPENCLAW_REMOTE_ACTIVATE_CMD:-}"
# Give OpenClaw more time to respond before treating turn 0 as a dead session.
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-240}"
# Lower the terminal reward weight so intermediate process signals matter more.
export PINCHBENCH_TERMINAL_REWARD_WEIGHT="${PINCHBENCH_TERMINAL_REWARD_WEIGHT:-0.3}"
# PRM self-judge 走 RunPod 本地 vLLM（和 agent 共享同一个模型）
export PRM_VLLM_BASE_URL="${PRM_VLLM_BASE_URL:-http://localhost:8000/v1}"
# Keep judge model name aligned with the served base model path to avoid
# vLLM 404s like "The model `Qwen3-4B` does not exist."
export PRM_MODEL="${PRM_MODEL:-${MODEL}}"
export PRM_API_KEY="${PRM_API_KEY:-dummy}"
# PinchBench grading judge: use DashScope qwen-plus if API key is available.
export PINCHBENCH_GRADE_JUDGE_MODEL="${PINCHBENCH_GRADE_JUDGE_MODEL:-qwen-plus}"
export PINCHBENCH_GRADE_JUDGE_BACKEND="${PINCHBENCH_GRADE_JUDGE_BACKEND:-api}"
export PINCHBENCH_GRADE_JUDGE_BASE_URL="${PINCHBENCH_GRADE_JUDGE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export PINCHBENCH_GRADE_JUDGE_API_KEY="${PINCHBENCH_GRADE_JUDGE_API_KEY:-${DASHSCOPE_API_KEY:-${JUDGE_API_KEY:-}}}"

# RunPod 常见坑：未 export OPENCLAW_HOST → 默认 localhost，SSH 预检连到本机而非 ECS，训练必歪或失败。
# 本地-only 调试：PINCHBENCH_ALLOW_LOCAL_OPENCLAW=1
_oc_host="${OPENCLAW_HOST:-localhost}"
if { [ "${_oc_host}" = "localhost" ] || [ "${_oc_host}" = "127.0.0.1" ]; } && [ "${PINCHBENCH_ALLOW_LOCAL_OPENCLAW:-0}" != "1" ]; then
    echo "ERROR: OPENCLAW_HOST 为 ${_oc_host}。在 RunPod 上请指向 ECS，例如:"
    echo "  export OPENCLAW_HOST=8.163.82.224"
    echo "若确要本机 OpenClaw 调试: PINCHBENCH_ALLOW_LOCAL_OPENCLAW=1 bash rl/train/run_reinforce_lora.sh"
    exit 1
fi

python3 -c "import os, sys; from pathlib import Path; sys.path.insert(0, str(Path('${REPO_ROOT}') / 'scripts')); from lib_grading import preflight_judge_connection; preflight_judge_connection(judge_model=os.environ.get('PINCHBENCH_GRADE_JUDGE_MODEL', 'qwen-plus'), judge_backend=os.environ.get('PINCHBENCH_GRADE_JUDGE_BACKEND', 'api'), judge_base_url=os.environ.get('PINCHBENCH_GRADE_JUDGE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'), judge_api_key=os.environ.get('PINCHBENCH_GRADE_JUDGE_API_KEY', ''))"

python3 - <<'PY'
import os
import subprocess

host = os.environ.get("OPENCLAW_HOST", "localhost")
user = os.environ.get("OPENCLAW_USER", "root")
port = os.environ.get("OPENCLAW_PORT", "22")
ssh_key = os.environ.get("OPENCLAW_SSH_KEY", "/root/.ssh/id_ed25519")
activate_cmd = os.environ.get(
    "OPENCLAW_REMOTE_ACTIVATE_CMD",
    "",
)
remote_cmd = "command -v openclaw >/dev/null && openclaw --version"
if activate_cmd.strip():
    remote_cmd = f"{activate_cmd.strip()} && {remote_cmd}"

print(f"Preflighting ECS OpenClaw: {user}@{host}:{port}")
result = subprocess.run(
    [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-i", ssh_key,
        "-p", str(port),
        f"{user}@{host}",
        remote_cmd,
    ],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print(result.stdout, end="")
    print(result.stderr, end="")
    raise SystemExit(f"ECS OpenClaw preflight failed with exit code {result.returncode}")
print(result.stdout.strip())
print("ECS OpenClaw preflight succeeded")
PY

if [ "${PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT:-0}" != "1" ]; then
python3 - <<'PY'
import json
import os
import subprocess

def parse_aliases(env_name: str, defaults: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(env_name, "").strip()
    if raw:
        aliases = [item.strip() for item in raw.split(",") if item.strip()]
        if aliases:
            return aliases
    return list(defaults)

def first_match(ready: set[str], aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in ready:
            return alias
    return None

host = os.environ.get("OPENCLAW_HOST", "localhost")
user = os.environ.get("OPENCLAW_USER", "root")
port = os.environ.get("OPENCLAW_PORT", "22")
ssh_key = os.environ.get("OPENCLAW_SSH_KEY", "/root/.ssh/id_ed25519")
activate_cmd = os.environ.get("OPENCLAW_REMOTE_ACTIVATE_CMD", "").strip()
remote_bin_dir = os.environ.get("OPENCLAW_REMOTE_BIN_DIR", "").strip()
remote_prefix = activate_cmd or (f'export PATH="{remote_bin_dir}:$PATH"' if remote_bin_dir else "")
remote_cmd = "openclaw skills list --eligible --json"
if remote_prefix:
    remote_cmd = f"{remote_prefix} && {remote_cmd}"

search_aliases = parse_aliases(
    "OPENCLAW_WEB_SEARCH_SKILLS",
    (
        "web_search",
        "ddg-search",
        "search-web",
        "web-search-free",
        "ddg-web-search",
        "dashscope-web-search",
        "local-web-search-skill",
        "websearch",
    ),
)
fetch_aliases = parse_aliases(
    "OPENCLAW_WEB_FETCH_SKILLS",
    (
        "web_fetch",
        "web-fetch",
        "web-fetch-markdown",
        "clean-web-fetch",
        "jina-web-fetcher",
        "safe-smart-web-fetch",
        "webfetch",
    ),
)

print(f"Preflighting ECS OpenClaw web skills: {user}@{host}:{port}")
result = subprocess.run(
    [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-i", ssh_key,
        "-p", str(port),
        f"{user}@{host}",
        remote_cmd,
    ],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    raise SystemExit(
        f"ECS OpenClaw web-skill preflight failed (exit {result.returncode}): "
        f"{result.stderr.strip() or result.stdout.strip()}"
    )

payload = json.loads(result.stdout)
ready = {
    item.get("name")
    for item in payload.get("skills", [])
    if isinstance(item, dict) and item.get("eligible") and not item.get("disabled")
}
search_match = first_match(ready, search_aliases)
fetch_match = first_match(ready, fetch_aliases)
print(f"Ready web search skill: {search_match or 'missing'}")
print(f"Ready web fetch skill: {fetch_match or 'missing'}")
if not search_match or not fetch_match:
    raise SystemExit(
        "ECS OpenClaw web-skill preflight failed: "
        f"missing required skill(s). ready={sorted(ready)} "
        f"search_aliases={search_aliases} fetch_aliases={fetch_aliases}. "
        "Set OPENCLAW_WEB_SEARCH_SKILLS / OPENCLAW_WEB_FETCH_SKILLS if you intentionally mapped equivalents."
    )
print("ECS OpenClaw web-skill preflight succeeded")
PY
fi

mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}"
# TensorBoard：veRL 读 TENSORBOARD_DIR（verl/utils/tracking.py）；须 pip install tensorboard

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${BATCH_SIZE}" \
    data.max_prompt_length=16384 \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${VLLM_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30 \
    actor_rollout_ref.rollout.agent.default_agent_loop=openclaw_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="${REWARD_MANAGER_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=pinchbench_rl \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    ${HYDRA_MAX_ACTOR_KEEP} \
    trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=20 \
    trainer.default_local_dir="${OUTPUT_DIR}"
