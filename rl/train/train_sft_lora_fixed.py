#!/usr/bin/env python3
"""SFT LoRA fine-tuning with robust assistant-token masking.

Fixes the old implementation that guessed assistant spans by regex over the
rendered chat template. This version builds labels by incrementally applying
the tokenizer chat template over message prefixes, then supervising only the
new tokens introduced by assistant messages.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _tokenize_messages(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _build_labeled_example(
    tokenizer,
    messages: list[dict[str, Any]],
    max_length: int,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, list[int]]:
    prev_text = ""
    assistant_spans: list[tuple[int, int]] = []
    final_text = ""

    for idx in range(len(messages)):
        prefix = messages[: idx + 1]
        curr_text = tokenizer.apply_chat_template(
            prefix,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        role = prefix[-1]["role"]

        if not curr_text.startswith(prev_text):
            common_chars = 0
            max_common = min(len(prev_text), len(curr_text))
            while common_chars < max_common and prev_text[common_chars] == curr_text[common_chars]:
                common_chars += 1
            logger.warning(
                "Chat template text prefix mismatch at message %d role=%s; fallback common chars=%d",
                idx,
                role,
                common_chars,
            )
            start_char = common_chars
        else:
            start_char = len(prev_text)

        end_char = len(curr_text)
        if role == "assistant" and end_char > start_char:
            assistant_spans.append((start_char, end_char))

        prev_text = curr_text
        final_text = curr_text

    tokens = tokenizer(
        final_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]
    labels = [-100] * len(input_ids)

    for i, (tok_start, tok_end) in enumerate(offsets):
        for span_start, span_end in assistant_spans:
            if tok_start >= span_start and tok_end <= span_end and tok_end > tok_start:
                labels[i] = input_ids[i]
                break

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def load_sft_data(data_path: str, tokenizer, max_length: int) -> Dataset:
    rows = [json.loads(line) for line in Path(data_path).read_text().splitlines() if line.strip()]
    logger.info("Loaded %d SFT rows from %s", len(rows), data_path)

    input_ids_list = []
    labels_list = []
    attention_masks = []
    supervised_counts = []

    for row_idx, item in enumerate(rows):
        if "messages" not in item:
            raise ValueError(f"Row {row_idx} missing messages")

        example = _build_labeled_example(
            tokenizer,
            item["messages"],
            max_length=max_length,
            tools=item.get("tools"),
        )
        supervised_tokens = sum(1 for x in example["labels"] if x != -100)

        input_ids_list.append(example["input_ids"])
        labels_list.append(example["labels"])
        attention_masks.append(example["attention_mask"])
        supervised_counts.append(supervised_tokens)

        if supervised_tokens == 0:
            logger.warning("Row %d has zero supervised assistant tokens", row_idx)

    logger.info(
        "SFT supervised token stats: min=%d max=%d avg=%.1f zero_rows=%d/%d",
        min(supervised_counts) if supervised_counts else 0,
        max(supervised_counts) if supervised_counts else 0,
        sum(supervised_counts) / len(supervised_counts) if supervised_counts else 0.0,
        sum(1 for x in supervised_counts if x == 0),
        len(supervised_counts),
    )

    return Dataset.from_dict(
        {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_masks,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save-strategy", type=str, default="epoch", choices=["no", "epoch"])
    args = parser.parse_args()

    logger.info("Loading model: %s", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_sft_data(args.data_path, tokenizer, args.max_length)
    logger.info("Dataset size: %d", len(dataset))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=args.bf16,
        logging_steps=1,
        save_strategy=args.save_strategy,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    logger.info("Starting SFT training...")
    trainer.train()

    logger.info("Saving LoRA adapter to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
