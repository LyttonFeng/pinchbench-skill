#!/usr/bin/env python3
"""DPO LoRA training for Qwen3-1.7B.

Current version focuses on train/infer parity:
1. Preserve the full prompt prefix before the first assistant turn, not just the first user message.
2. Render prompt/chosen/rejected with tokenizer.apply_chat_template(..., tools=tools) so
   DPO sees the same system+tool schema family as runtime.
3. Keep only the first assistant turn as completion to avoid length bias from tool results.
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


def _render_prompt_and_completion(tokenizer, prompt_msgs, completion_msgs, tools=None) -> tuple[str, str]:
    """Render prompt/completion text with chat template + tools parity."""
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_msgs + completion_msgs,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("Chat template mismatch: full_text does not start with prompt_text")
    return prompt_text, full_text[len(prompt_text):]


def _truncate_prompt_for_completion(
    tokenizer,
    prompt_text: str,
    chosen_text: str,
    rejected_text: str,
    *,
    max_length: int | None,
    max_prompt_length: int | None,
    prompt_truncation_mode: str,
) -> tuple[str, str, str]:
    """Truncate prompt at token level while preserving the full completion."""
    if max_length is None:
        return prompt_text, chosen_text, rejected_text

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(chosen_text, add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(rejected_text, add_special_tokens=False)["input_ids"]

    completion_budget = max(len(chosen_ids), len(rejected_ids))
    available_prompt = max_length - completion_budget
    if available_prompt <= 0:
        raise ValueError(
            f"Completion too long for max_length={max_length}: "
            f"chosen={len(chosen_ids)}, rejected={len(rejected_ids)}"
        )

    if max_prompt_length is not None:
        available_prompt = min(available_prompt, max_prompt_length)

    if len(prompt_ids) <= available_prompt:
        return prompt_text, chosen_text, rejected_text

    if prompt_truncation_mode == "keep_start":
        truncated_prompt_ids = prompt_ids[:available_prompt]
    elif prompt_truncation_mode == "keep_end":
        truncated_prompt_ids = prompt_ids[-available_prompt:]
    else:
        raise ValueError(
            f"Unsupported prompt_truncation_mode={prompt_truncation_mode}; expected keep_start or keep_end"
        )

    truncated_prompt_text = tokenizer.decode(
        truncated_prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return truncated_prompt_text, chosen_text, rejected_text


def load_dpo_dataset(
    data_path: str,
    tokenizer,
    *,
    max_length: int | None,
    max_prompt_length: int | None,
    prompt_truncation_mode: str,
) -> Dataset:
    """Load DPO pairs and pre-render them for TRL DPOTrainer."""
    rows = [json.loads(l) for l in Path(data_path).read_text().splitlines() if l.strip()]
    logger.info("Loaded %d DPO pairs from %s", len(rows), data_path)

    prompts, chosens, rejecteds = [], [], []
    prompt_lens, chosen_lens, rejected_lens = [], [], []
    for row in rows:
        chosen_msgs = row["chosen"].get("messages", [])
        rejected_msgs = row["rejected"].get("messages", [])
        tools = row.get("tools") or row["chosen"].get("tools") or row["rejected"].get("tools")

        if not chosen_msgs or not rejected_msgs:
            logger.warning("Skipping row with empty messages: %s", row.get("variant_id"))
            continue

        # Keep the full prompt prefix up to the first assistant message.
        first_assistant_idx = next((i for i, m in enumerate(chosen_msgs) if m["role"] == "assistant"), None)
        rejected_first_assistant_idx = next((i for i, m in enumerate(rejected_msgs) if m["role"] == "assistant"), None)
        if first_assistant_idx is None or rejected_first_assistant_idx is None:
            logger.warning("Skipping row with no assistant response: %s", row.get("variant_id"))
            continue
        prompt_msgs = chosen_msgs[:first_assistant_idx]
        if not prompt_msgs:
            logger.warning("Skipping row with empty prompt prefix: %s", row.get("variant_id"))
            continue

        # CRITICAL FIX: Only use first assistant turn (tool_call)
        # Don't include tool result to avoid length bias from binary garbage
        chosen_completion = [chosen_msgs[first_assistant_idx]]
        rejected_completion = [rejected_msgs[rejected_first_assistant_idx]]

        try:
            prompt_text, chosen_text = _render_prompt_and_completion(
                tokenizer, prompt_msgs, chosen_completion, tools=tools
            )
            _, rejected_text = _render_prompt_and_completion(
                tokenizer, prompt_msgs, rejected_completion, tools=tools
            )
            prompt_text, chosen_text, rejected_text = _truncate_prompt_for_completion(
                tokenizer,
                prompt_text,
                chosen_text,
                rejected_text,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                prompt_truncation_mode=prompt_truncation_mode,
            )
        except Exception as exc:
            logger.warning("Skipping row %s due to render error: %s", row.get("variant_id"), exc)
            continue

        prompts.append(prompt_text)
        chosens.append(chosen_text)
        rejecteds.append(rejected_text)
        prompt_lens.append(len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]))
        chosen_lens.append(len(tokenizer(chosen_text, add_special_tokens=False)["input_ids"]))
        rejected_lens.append(len(tokenizer(rejected_text, add_special_tokens=False)["input_ids"]))

    logger.info("Built %d valid DPO samples", len(prompts))
    if prompts:
        logger.info(
            "Token lengths after truncation: prompt min=%d max=%d avg=%.1f | "
            "chosen min=%d max=%d avg=%.1f | rejected min=%d max=%d avg=%.1f",
            min(prompt_lens), max(prompt_lens), sum(prompt_lens) / len(prompt_lens),
            min(chosen_lens), max(chosen_lens), sum(chosen_lens) / len(chosen_lens),
            min(rejected_lens), max(rejected_lens), sum(rejected_lens) / len(rejected_lens),
        )
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
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--prompt-truncation-mode", choices=["keep_start", "keep_end"], default="keep_end")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dpo_dataset(
        args.data_path,
        tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        prompt_truncation_mode=args.prompt_truncation_mode,
    )
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
