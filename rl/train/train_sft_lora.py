#!/usr/bin/env python3
"""SFT LoRA fine-tuning for activating exec tool usage on task_18_spreadsheet_summary."""
import argparse
import json
import subprocess
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling


def cleanup_gpu_memory():
    """Clean up GPU memory before training."""
    print("🧹 Cleaning GPU memory...")
    subprocess.run("pkill -9 python3 || true", shell=True)
    subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9 || true", shell=True)
    import time
    time.sleep(3)
    print("✅ GPU memory cleaned")


def load_sft_data(data_path: str, tokenizer, max_length: int) -> Dataset:
    """Load and tokenize SFT data, masking non-assistant tokens."""
    with open(data_path) as f:
        data = [json.loads(line) for line in f]

    input_ids_list = []
    labels_list = []

    for item in data:
        # Format with chat template
        text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize with return_offsets_mapping to track char positions
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True
        )
        input_ids = tokens['input_ids']
        offsets = tokens['offset_mapping']

        # Create labels: mask everything except assistant responses
        # Find all assistant response regions in text
        labels = [-100] * len(input_ids)

        # Find all <|im_start|>assistant ... <|im_start|> regions
        import re
        pattern = r'<\|im_start\|>assistant\n(.*?)(?=<\|im_start\||$)'
        for match in re.finditer(pattern, text, re.DOTALL):
            start_char = match.start()
            end_char = match.end()

            # Find tokens that overlap with this region
            for i, (token_start, token_end) in enumerate(offsets):
                if token_start >= start_char and token_end <= end_char:
                    labels[i] = input_ids[i]

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": [[1] * len(ids) for ids in input_ids_list]
    })


def main():
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
    args = parser.parse_args()

    # CRITICAL: Clean GPU memory before loading model
    cleanup_gpu_memory()

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup LoRA
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

    # Load data with manual label masking
    print(f"Loading data: {args.data_path}")
    dataset = load_sft_data(args.data_path, tokenizer, args.max_length)
    print(f"Dataset size: {len(dataset)}")

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=args.bf16,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Use standard Trainer with DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting SFT training...")
    trainer.train()

    # Save
    print(f"Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
