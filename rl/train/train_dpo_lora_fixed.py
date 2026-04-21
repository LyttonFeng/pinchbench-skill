#!/usr/bin/env python3
"""DPO LoRA training for Qwen3-1.7B - FIXED VERSION.

Key fixes:
1. Use messages format directly (don't serialize tool_calls to text)
2. Keep tool role as-is (don't convert to user)
3. Let TRL handle prompt/completion split
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_dpo_dataset(data_path: str, tokenizer) -> Dataset:
    """Load DPO pairs and format for TRL DPOTrainer.

    TRL expects:
    - prompt: list of messages (user prompt only)
    - chosen: list of messages (completion only, WITHOUT prompt)
    - rejected: list of messages (completion only, WITHOUT prompt)

    TRL will concatenate prompt + chosen/rejected internally.
    """
    rows = [json.loads(l) for l in Path(data_path).read_text().splitlines() if l.strip()]
    logger.info("Loaded %d DPO pairs from %s", len(rows), data_path)

    prompts, chosens, rejecteds = [], [], []
    for row in rows:
        chosen_msgs = row["chosen"].get("messages", [])
        rejected_msgs = row["rejected"].get("messages", [])

        if not chosen_msgs or not rejected_msgs:
            logger.warning("Skipping row with empty messages: %s", row.get("variant_id"))
            continue

        # Extract prompt (first user message)
        if not chosen_msgs or chosen_msgs[0]["role"] != "user":
            logger.warning("Skipping row with no user prompt: %s", row.get("variant_id"))
            continue

        prompt_msgs = [chosen_msgs[0]]

        # IMPORTANT: chosen/rejected should NOT include the prompt
        # TRL will concatenate prompt + chosen/rejected internally
        chosen_completion = chosen_msgs[1:]  # Skip first user message
        rejected_completion = rejected_msgs[1:]  # Skip first user message

        if not chosen_completion or not rejected_completion:
            logger.warning("Skipping row with empty completions: %s", row.get("variant_id"))
            continue

        prompts.append(prompt_msgs)
        chosens.append(chosen_completion)
        rejecteds.append(rejected_completion)

    logger.info("Built %d valid DPO samples", len(prompts))
    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dpo_dataset(args.data_path, tokenizer)
    logger.info("Dataset size: %d", len(dataset))

    logger.info("Loading model: %s", args.model_name)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    # Use ref_model=None to share reference with model (saves memory)
    ref_model = None

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=args.bf16,
        fp16=False,
        beta=args.beta,
        max_length=args.max_length,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        dataset_num_proc=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    logger.info("Saving LoRA adapter to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
