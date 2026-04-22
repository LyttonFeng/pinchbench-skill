#!/usr/bin/env python3
"""Debug TRL tokenization to understand the mismatch warnings."""

import json
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Load one sample
with open("rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_pure_focused_train.jsonl") as f:
    sample = json.loads(f.readline())

print("=" * 80)
print("SAMPLE STRUCTURE")
print("=" * 80)
print(f"Keys: {sample.keys()}")
print(f"Prompt type: {type(sample['prompt'])}")
print(f"Chosen type: {type(sample['chosen'])}")
print(f"Rejected type: {type(sample['rejected'])}")

# Extract messages
prompt_messages = sample['prompt']
chosen_messages = sample['chosen']
rejected_messages = sample['rejected']

print("\n" + "=" * 80)
print("MESSAGE COUNTS")
print("=" * 80)
print(f"Prompt messages: {len(prompt_messages)}")
print(f"Chosen messages: {len(chosen_messages)}")
print(f"Rejected messages: {len(rejected_messages)}")

print("\n" + "=" * 80)
print("LAST PROMPT MESSAGE")
print("=" * 80)
print(json.dumps(prompt_messages[-1], indent=2))

print("\n" + "=" * 80)
print("FIRST CHOSEN MESSAGE (should be assistant)")
print("=" * 80)
print(json.dumps(chosen_messages[0], indent=2))

print("\n" + "=" * 80)
print("FIRST REJECTED MESSAGE (should be assistant)")
print("=" * 80)
print(json.dumps(rejected_messages[0], indent=2))

# Now tokenize like TRL does
print("\n" + "=" * 80)
print("TRL TOKENIZATION SIMULATION")
print("=" * 80)

# TRL tokenizes prompt
prompt_tokens = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize=True,
    add_generation_prompt=False  # TRL doesn't add generation prompt for prompt
)

# TRL tokenizes prompt + chosen
full_chosen_messages = prompt_messages + chosen_messages
chosen_tokens = tokenizer.apply_chat_template(
    full_chosen_messages,
    tokenize=True,
    add_generation_prompt=False
)

# TRL tokenizes prompt + rejected
full_rejected_messages = prompt_messages + rejected_messages
rejected_tokens = tokenizer.apply_chat_template(
    full_rejected_messages,
    tokenize=True,
    add_generation_prompt=False
)

print(f"Prompt tokens: {len(prompt_tokens)}")
print(f"Chosen tokens: {len(chosen_tokens)}")
print(f"Rejected tokens: {len(rejected_tokens)}")

# Check if prompt tokens match the start of chosen/rejected
prompt_matches_chosen = chosen_tokens[:len(prompt_tokens)] == prompt_tokens
prompt_matches_rejected = rejected_tokens[:len(prompt_tokens)] == rejected_tokens

print(f"\nPrompt matches start of chosen: {prompt_matches_chosen}")
print(f"Prompt matches start of rejected: {prompt_matches_rejected}")

if not prompt_matches_chosen:
    print("\n" + "=" * 80)
    print("MISMATCH DETAILS (CHOSEN)")
    print("=" * 80)
    print(f"First 20 prompt tokens: {prompt_tokens[:20]}")
    print(f"First 20 chosen tokens: {chosen_tokens[:20]}")

    # Find first mismatch
    for i in range(min(len(prompt_tokens), len(chosen_tokens))):
        if prompt_tokens[i] != chosen_tokens[i]:
            print(f"\nFirst mismatch at position {i}:")
            print(f"  Prompt token: {prompt_tokens[i]} -> '{tokenizer.decode([prompt_tokens[i]])}'")
            print(f"  Chosen token: {chosen_tokens[i]} -> '{tokenizer.decode([chosen_tokens[i]])}'")
            break

# Decode to see the actual text
print("\n" + "=" * 80)
print("DECODED TEXT")
print("=" * 80)
print("Prompt text (last 200 chars):")
prompt_text = tokenizer.decode(prompt_tokens)
print(prompt_text[-200:])

print("\nChosen text (first 200 chars after prompt):")
chosen_text = tokenizer.decode(chosen_tokens)
print(chosen_text[len(prompt_text):len(prompt_text)+200])

print("\nRejected text (first 200 chars after prompt):")
rejected_text = tokenizer.decode(rejected_tokens)
print(rejected_text[len(prompt_text):len(prompt_text)+200])
