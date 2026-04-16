# RL Training v2 Config

**Date:** 2026-04-16
**Run:** RUN_VERSION=v2 → checkpoint dir: `rl/checkpoints/reinforce_lora_v2/`

## 训练超参

| 参数 | 值 | 说明 |
|---|---|---|
| BATCH_SIZE | 4 | 每步4个episode（v1是2，太小导致梯度不稳） |
| TOTAL_EPOCHS | 4 | 8 tasks / 4 batch = 2 steps/epoch × 4 epochs = **8 total steps** |
| TEST_FREQ | 2 | 每2步跑val + 记录指标 |
| SAVE_FREQ | 2 | 每2步保存checkpoint（与TEST_FREQ一致，best_ckpt要求） |
| PINCHBENCH_LORA_ONLY_CKPT | 1 | 只保存LoRA adapter (~50MB)，不保存FSDP完整权重(~17GB) |
| PINCHBENCH_TASK_EMA_INIT | 0.3 | per-task EMA baseline初始值（对齐terminal_reward_weight=0.5） |
| TRAINER_RESUME_MODE | disable | 从头训练，不续训 |

## 算法参数（代码内）

| 参数 | 值 | 说明 |
|---|---|---|
| algorithm.adv_estimator | reinforce_plus_plus | REINFORCE++ |
| algorithm.use_kl_in_reward | True | KL惩罚加入reward（v1是False，改进）|
| actor.use_kl_loss | True | KL作为额外loss项（kl_coef=0.05） |
| actor.kl_loss_coef | 0.05 | KL loss系数（reward侧kl_coef=0.001，极小，影响可忽略） |
| actor.ppo_epochs | 1 | 严格on-policy |
| actor.shuffle | False | 配合round-robin数据集 |
| MICRO_BATCH | 1 | 每GPU micro batch=1，防OOM |
| LR | 2e-5 | 学习率 |
| LORA_RANK | 32 | LoRA rank |
| LORA_ALPHA | 64 | LoRA alpha |

## 奖励参数

| 参数 | 值 | 说明 |
|---|---|---|
| REWARD_MODE | oracle-judge | qwen-plus逐turn打分 |
| PINCHBENCH_TERMINAL_REWARD_WEIGHT | 0.5 | 终端成功权重（v1是0.3，加强任务完成信号） |
| PINCHBENCH_REWARD_RETURN_MODE | turn | turn-level reward（放到<\|im_end\|> token） |

## 环境

| 参数 | 值 |
|---|---|
| OPENCLAW_HOST | 8.163.82.224 (ECS) |
| OPENCLAW_PORT | 22 |
| OPENCLAW_USER | root |
| VLLM_GPU_MEM_UTIL | 0.28 |
| MAX_TURNS | 16 |
| Model | Qwen/Qwen3-4B |

## v1→v2 改进点总结

1. **BATCH_SIZE 2→4**：降低advantage方差，梯度更稳定
2. **use_kl_in_reward False→True**：防止后期模型漂离reference policy
3. **TASK_EMA_INIT 0.5→0.3**：与max reward对齐，避免早期advantage全负
4. **TOTAL_STEPS 32→8**：v1 step16翻车，8步足够看到效果
5. **LORA_ONLY_CKPT 0→1**：避免MFS大文件写入失败

## Baseline对比

| Run | Model | Score |
|---|---|---|
| 0057 | Qwen3-4B base | 50.4% |
| 0078 | v1 step8 LoRA | **66.0%** |
| 0077 | v1 step17 LoRA | 51.6%（过训练退化） |
