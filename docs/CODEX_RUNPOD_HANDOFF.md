# Codex / 协作者接手说明：RunPod、训练、磁盘与 Fork

本文档给 **Codex 或其它助手** 读：如何连 RunPod、仓库在哪、训练怎么起、checkpoint/满盘怎么排、本仓库相关改动在哪。

**Fork 远程（推送用这个，不是 org 的 origin）：**

```bash
git remote add fork git@github.com:LyttonFeng/pinchbench-skill.git   # 若已存在则跳过
git push fork main
```

---

## 1. 连接 RunPod（不要在仓库里提交真实 IP/端口）

1. RunPod 控制台打开对应 Pod → **Connect**。
2. **SSH over exposed TCP**：会给出 `ssh root@<公网IP> -p <端口> -i ~/.ssh/<你的密钥>`。端口映射到容器内 **22**；支持 SCP/SFTP。
3. **Web Terminal**：浏览器里直接进 shell（无需 SSH）。
4. Pod 重建 / 新 Pod 后 **IP 和端口会变**，以控制台为准；不要把当前会话的 IP 写进 git 里的明文配置。

本机示例（把占位符换成控制台里的值）：

```bash
ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519
```

---

## 2. Pod 上的路径约定（与 `rl/scripts/setup_new_pod.sh` 一致）

| 项 | 路径 |
|----|------|
| 仓库 clone | `/workspace/pinchbench-skill`（`REPO_DIR`） |
| Hugging Face 缓存（建议） | `/workspace/hf_cache`（`HF_HOME` / `HF_HUB_CACHE` 常指到这里） |
| 训练 checkpoint 根目录 | `/workspace/pinchbench-skill/rl/checkpoints/reinforce_lora` |
| TensorBoard 日志（默认） | 同上目录下 `tensorboard/` |

**磁盘**：重点看 **`df -h /workspace`**。系统根盘 `overlay` 通常很小；持久数据应在 `/workspace`。

---

## 3. 训练命令

在 Pod 里：

```bash
cd /workspace/pinchbench-skill
export OPENCLAW_HOST=...   # 见 setup_new_pod / run 脚本注释
export OPENCLAW_USER=root
# self-judge / oracle 等按需
export JUDGE_API_KEY=...
bash rl/train/run_reinforce_lora.sh
# 或 tee 日志:  bash rl/train/run_reinforce_lora.sh 2>&1 | tee /workspace/train.log
```

依赖：`verl`、`vllm`、`transformers`、`peft` 等（见脚本头注释）。

---

## 4. Checkpoint 结构（为何占几十 GB）

每次保存一个目录：`rl/checkpoints/reinforce_lora/global_step_<N>/`

- **`actor/`**：FSDP 分片 + optimizer，**体积最大（常单目录 ~10–20GB 级）**。
- **`actor/lora_adapter/`**：`adapter_model.safetensors` + `adapter_config.json`，**推理主要用这个（约百 MB）**。
- **`data.pt`**：dataloader 状态，很小。
- **`latest_checkpointed_iteration.txt`**：veRL 记录的最近保存步。
- **`best_ckpt_state.json`**：仅当启用「按验证集只留最佳」时存在（见下节）。

**满盘 / 保存失败**：PyTorch 可能出现 `inline_container.cc` / `unexpected pos` 一类 zip 写入错误，**常见真实原因是磁盘满或 inode**，不是逻辑 bug。

**损坏的 checkpoint**：若某 `global_step_*` 的 `du -sh` **异常小**（例如几十 MB）且目录里只有零散 `.pt`、没有完整 `actor/lora_adapter` 结构，视为**写盘中断**；不要用「步数最大」脚本盲目保留它——应保留 **体积正常、结构完整** 的那一阶，或依赖 `best_ckpt_state.json`。

---

## 5. 本仓库里的磁盘与「只留最佳」机制

### 5.1 手动清理旧 checkpoint（在 Pod 上执行）

`rl/scripts/prune_old_ckpts.sh`：

- 若存在 **`best_ckpt_state.json`**：只保留其中的 `best_step` 对应目录。
- 否则：保留步数最大的 **`KEEP_LATEST_N`** 个（默认 1）。**注意**：若「最大步」目录是损坏的小目录，应人工判断或先删坏目录再跑脚本。

```bash
cd /workspace/pinchbench-skill
DRY_RUN=1 bash rl/scripts/prune_old_ckpts.sh
bash rl/scripts/prune_old_ckpts.sh
# 非默认路径:
# CKPT_ROOT=/workspace/.../rl/checkpoints/reinforce_lora bash rl/scripts/prune_old_ckpts.sh
```

### 5.2 训练时自动「只保留验证最佳」

- 环境变量 **`PINCHBENCH_BEST_CKPT=1`**（`run_reinforce_lora.sh` 默认开启）。
- 机制：仓库根目录 **`sitecustomize.py`** 在 Python 启动时加载（需 **`PYTHONPATH` 含仓库根**，脚本已 `export PYTHONPATH="${REPO_ROOT}:..."`），导入 **`rl/verl_best_ckpt_patch.py`**，monkey-patch `verl` 的 `RayPPOTrainer._validate`：每次验证后用 **`val-core/.../reward/mean@*`**（越高越好）比较，删非最佳 `global_step_*`。
- 关闭：`PINCHBENCH_BEST_CKPT=0 bash rl/train/run_reinforce_lora.sh`。
- **Ray worker** 需继承同一 `PYTHONPATH` / 环境变量（当前启动方式下与 driver 一致即可）。

### 5.3 `run_reinforce_lora.sh` 中与 veRL 的配合

- `PINCHBENCH_BEST_CKPT=1` 时：`trainer.max_actor_ckpt_to_keep=null`，避免与自定义删目录冲突；**`SAVE_FREQ` 默认与 `TEST_FREQ` 对齐**，保证每个验证步都有新 checkpoint 可参与比较。
- `TEST_FREQ` 可通过环境变量覆盖（默认 5）。

---

## 6. `/workspace` 空间都去哪了（典型）

一次排查里常见占比：

- **`rl/checkpoints/.../global_step_*`**：每个完整步可 **~10–20GB**。
- **`/workspace/hf_cache`**：基座模型（如 Qwen3-4B）**约十余 GB**。

两者叠加很容易到 **30GB+**，属预期；不是「隐藏垃圾文件」。

---

## 7. OpenClaw / 判分（上下文）

- Agent 跑在 **阿里云 ECS**，训练在 RunPod；`OPENCLAW_HOST` 等指向 ECS（`setup_new_pod.sh` 与脚本注释中有 IP/用户说明）。
- Reward 模式：`REWARD_MODE` = `baseline` / `rule` / `self-judge` / `oracle-judge` 等（见 `run_reinforce_lora.sh` 头注释）。

### 7.1 最近一次可恢复快照

当前能稳定推进到 vLLM/Ray 初始化的配置基线是：

- Repo HEAD：`f34aaec`（`Reduce vLLM max model length`）
- 训练入口：`rl/train/run_reinforce_lora.sh`
- 模型：`Qwen/Qwen3-4B`
- LoRA：`rank=32`, `alpha=64`
- 单卡 actor：`actor_rollout_ref.actor.fsdp_config.param_offload=True`
- rollout：
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.30`
  - `actor_rollout_ref.rollout.max_model_len=19456`
  - `actor_rollout_ref.rollout.tensor_model_parallel_size=1`
- judge：
  - `PINCHBENCH_GRADE_JUDGE_BACKEND=api`
  - `PINCHBENCH_GRADE_JUDGE_MODEL=qwen-plus`
  - `PINCHBENCH_GRADE_JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- OpenClaw：
  - ECS 侧 `openclaw` 应直接在 PATH 上可用，不依赖 `coursebot`
  - web-heavy 任务必须通过 preflight
  - `ddg-search` 作为 `web_search` 别名
  - `web-fetch` 作为 `web_fetch` 别名

重启 pod 后先做：

```bash
cd /workspace/pinchbench-skill
git pull --ff-only origin main
git rev-parse --short HEAD
```

---

## 8. 安全

- **勿**将 RunPod SSH 命令、私钥、API Key 提交进 git。
- 若在公开渠道泄露过连接信息，应轮换密钥或重建 Pod。

---

## 9. 相关文件清单（便于 Codex 搜代码）

| 文件 | 作用 |
|------|------|
| `rl/train/run_reinforce_lora.sh` | 训练入口、checkpoint/val 频率、`PYTHONPATH`、`PINCHBENCH_BEST_CKPT` |
| `sitecustomize.py` | 条件加载 best-ckpt patch |
| `rl/verl_best_ckpt_patch.py` | veRL `_validate` 后按 val 指标删 checkpoint |
| `rl/scripts/prune_old_ckpts.sh` | Pod 上手动释放磁盘 |
| `rl/scripts/setup_new_pod.sh` | RunPod 初始化、常见坑（含磁盘满） |
| `rl/scripts/start_tensorboard.sh` | TensorBoard（若存在） |

将此文件与上述脚本一并 **`git push fork main`** 后，Codex 拉取 fork 即可对齐上下文。
