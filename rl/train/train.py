"""
veRL single-sample PG 训练脚本。

算法：turn-level clipped policy gradient + critic baseline + reference-policy KL
参考：rl/README.md 和 docs/rl-algorithm.md（Phase 1）

用法（RunPod A100 上执行）：
    python rl/train/train.py \
        --data rl/data/samples_rescored.jsonl \
        --model Qwen/Qwen3-1.7B \
        --output rl/checkpoints/

依赖：
    pip install verl transformers torch

注意：
    - 需要先运行 rescore.py 补全 logprobs
    - 只使用 split=train 的样本训练
    - automated grading 的 task 优先（reward 无噪声）
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# rl/ 目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from schema import TrainingSample  # type: ignore
from data import load_samples, sample_to_verl  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="veRL single-sample PG 训练")
    parser.add_argument("--data", type=Path, required=True, help="rescored JSONL 文件")
    parser.add_argument("--model", required=True, help="模型路径或 HuggingFace ID")
    parser.add_argument("--output", type=Path, default=Path("rl/checkpoints"), help="checkpoint 输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练 epoch 数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--kl-coef", type=float, default=0.1, help="reference KL 系数 β")
    parser.add_argument("--critic-coef", type=float, default=0.5, help="critic loss 系数")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--grading-types",
        nargs="+",
        default=["automated"],
        help="只使用指定 grading_type 的样本（默认只用 automated）",
    )
    parser.add_argument("--no-critic", action="store_true", help="去掉 critic（REINFORCE 对照实验）")
    parser.add_argument("--save-every", type=int, default=100, help="每 N step 保存一次 checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalize_advantages(rewards: list[float]) -> list[float]:
    """对 reward 做标准化，作为 advantage（无 critic 时）。"""
    if len(rewards) <= 1:
        return [0.0] * len(rewards)
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var + 1e-8)
    return [(r - mu) / std for r in rewards]


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        logger.error("缺少依赖: %s\n请运行: pip install torch transformers peft", e)
        sys.exit(1)

    # 加载训练数据
    samples = load_samples(
        args.data,
        split="train",
        require_logprobs=True,
        grading_types=args.grading_types,
    )
    if not samples:
        logger.error("没有可用的训练样本，检查数据文件和过滤条件")
        sys.exit(1)

    # 转成 step list
    all_steps: list[dict[str, Any]] = []
    for sample in samples:
        all_steps.extend(sample_to_verl(sample))
    logger.info("共 %d 个训练 step（来自 %d 条样本）", len(all_steps), len(samples))

    # 加载模型
    logger.info("加载模型: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Reference model（冻结，用于 KL 正则）
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Critic head（共享 backbone，Linear 输出标量 V）
    critic_head = None
    if not args.no_critic:
        hidden_size = model.config.hidden_size
        critic_head = torch.nn.Linear(hidden_size, 1, bias=False)
        logger.info("Critic head 已启用（hidden_size=%d）", hidden_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ref_model = ref_model.to(device)
    if critic_head:
        critic_head = critic_head.to(device)

    # Optimizer
    params = list(model.parameters())
    if critic_head:
        params += list(critic_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    args.output.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        import random
        random.shuffle(all_steps)
        epoch_loss = 0.0

        for step_data in all_steps:
            prompt_text = step_data["prompt_text"]
            response_text = step_data["response_text"]
            logprobs_old = step_data["logprobs_old"]
            reward = step_data["reward"]

            # Tokenize
            full_text = prompt_text + "\n<|assistant|>\n" + response_text
            prompt_enc = tokenizer(prompt_text, return_tensors="pt").to(device)
            full_enc = tokenizer(full_text, return_tensors="pt").to(device)

            prompt_len = prompt_enc["input_ids"].shape[1]
            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]

            # Forward（当前策略）
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=(critic_head is not None),
            )
            logits = outputs.logits  # [1, seq_len, vocab_size]

            # 计算 response 部分的 logprobs（新策略）
            response_ids = input_ids[0, prompt_len:]
            response_logits = logits[0, prompt_len - 1 : -1]  # shifted
            log_probs_new = torch.nn.functional.log_softmax(response_logits, dim=-1)
            selected_logprobs_new = log_probs_new[
                torch.arange(len(response_ids)), response_ids
            ]  # [resp_len]

            # Old logprobs（re-score 时记录的）
            n = min(len(logprobs_old), len(selected_logprobs_new))
            if n == 0:
                continue
            logprobs_old_t = torch.tensor(logprobs_old[:n], dtype=torch.float32, device=device)
            logprobs_new_t = selected_logprobs_new[:n]

            # Importance ratio
            rho = torch.exp(logprobs_new_t - logprobs_old_t)

            # Advantage
            if critic_head and not args.no_critic:
                # h_t: 最后一个 response token 的 hidden state
                hidden = outputs.hidden_states[-1][0, prompt_len + n - 1]
                v_t = critic_head(hidden).squeeze()
                advantage = reward - v_t.detach().item()
                # Critic loss
                loss_critic = 0.5 * (v_t - reward) ** 2
            else:
                advantage = reward  # REINFORCE（无 baseline）
                loss_critic = torch.tensor(0.0, device=device)

            advantage_t = torch.tensor(advantage, dtype=torch.float32, device=device)

            # Clipped PG loss
            loss_pg = -torch.mean(
                torch.min(
                    rho * advantage_t,
                    torch.clamp(rho, 1 - args.clip_eps, 1 + args.clip_eps) * advantage_t,
                )
            )

            # Reference KL
            with torch.no_grad():
                ref_outputs = ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                ref_logits = ref_outputs.logits[0, prompt_len - 1 : -1]
                ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                ref_selected = ref_log_probs[torch.arange(len(response_ids)), response_ids][:n]

            kl = torch.mean(logprobs_new_t - ref_selected)

            # 联合 loss
            loss = loss_pg + args.kl_coef * kl + args.critic_coef * loss_critic

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                logger.info(
                    "epoch=%d step=%d loss=%.4f pg=%.4f kl=%.4f reward=%.3f adv=%.3f",
                    epoch + 1,
                    global_step,
                    loss.item(),
                    loss_pg.item(),
                    kl.item(),
                    reward,
                    advantage if isinstance(advantage, float) else advantage_t.item(),
                )

            if global_step % args.save_every == 0:
                ckpt_dir = args.output / f"step-{global_step:06d}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                if critic_head:
                    import torch as _torch
                    _torch.save(critic_head.state_dict(), ckpt_dir / "critic_head.pt")
                logger.info("保存 checkpoint → %s", ckpt_dir)

        avg_loss = epoch_loss / max(len(all_steps), 1)
        logger.info("epoch %d 完成，平均 loss=%.4f", epoch + 1, avg_loss)

    # 保存最终 checkpoint
    final_dir = args.output / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if critic_head:
        import torch as _torch
        _torch.save(critic_head.state_dict(), final_dir / "critic_head.pt")
    logger.info("训练完成，最终 checkpoint → %s", final_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)
