#!/usr/bin/env python3
"""Debug what TRL actually sees during DPO training."""

import json
from pathlib import Path
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load first sample
data_path = Path("rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_pure_focused_train.jsonl")
row = json.loads(data_path.read_text().splitlines()[1])  # Use sample 1 (sample 0 has mismatched prompt)

chosen_msgs = row["chosen"]["messages"]
rejected_msgs = row["rejected"]["messages"]

print("=" * 80)
print("SAMPLE 1 STRUCTURE")
print("=" * 80)

print("\nChosen messages:")
for i, msg in enumerate(chosen_msgs):
    print(f"  {i}. role={msg['role']:10s} has_tool_calls={'tool_calls' in msg}")
    if 'tool_calls' in msg and msg['tool_calls']:
        print(f"     -> tool: {msg['tool_calls'][0]['function']['name']}")

print("\nRejected messages:")
for i, msg in enumerate(rejected_msgs):
    print(f"  {i}. role={msg['role']:10s} has_tool_calls={'tool_calls' in msg}")
    if 'tool_calls' in msg and msg['tool_calls']:
        print(f"     -> tool: {msg['tool_calls'][0]['function']['name']}")

# Simulate what train_dpo_lora_fixed.py does
prompt_msgs = [chosen_msgs[0]]
chosen_completion = chosen_msgs[1:]
rejected_completion = rejected_msgs[1:]

print("\n" + "=" * 80)
print("WHAT TRL RECEIVES (after train_dpo_lora_fixed.py processing)")
print("=" * 80)

print(f"\nPrompt: {len(prompt_msgs)} messages")
print(f"Chosen completion: {len(chosen_completion)} messages")
print(f"Rejected completion: {len(rejected_completion)} messages")

# Apply chat template
print("\n" + "=" * 80)
print("TOKENIZED SEQUENCES")
print("=" * 80)

prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=False)
print(f"\nPrompt text ({len(prompt_text)} chars):")
print(prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text)

# TRL does: tokenize(prompt + chosen)
full_chosen_text = tokenizer.apply_chat_template(
    prompt_msgs + chosen_completion,
    tokenize=False,
    add_generation_prompt=False
)
full_rejected_text = tokenizer.apply_chat_template(
    prompt_msgs + rejected_completion,
    tokenize=False,
    add_generation_prompt=False
)

print(f"\nFull chosen text ({len(full_chosen_text)} chars):")
print(full_chosen_text[:400] + "..." if len(full_chosen_text) > 400 else full_chosen_text)

print(f"\nFull rejected text ({len(full_rejected_text)} chars):")
print(full_rejected_text[:400] + "..." if len(full_rejected_text) > 400 else full_rejected_text)

# Tokenize
prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
chosen_tokens = tokenizer.encode(full_chosen_text, add_special_tokens=False)
rejected_tokens = tokenizer.encode(full_rejected_text, add_special_tokens=False)

print("\n" + "=" * 80)
print("TOKEN COUNTS")
print("=" * 80)
print(f"Prompt: {len(prompt_tokens)} tokens")
print(f"Full chosen: {len(chosen_tokens)} tokens")
print(f"Full rejected: {len(rejected_tokens)} tokens")
print(f"Chosen completion: {len(chosen_tokens) - len(prompt_tokens)} tokens")
print(f"Rejected completion: {len(rejected_tokens) - len(prompt_tokens)} tokens")

# Check if prompt matches
prompt_matches_chosen = chosen_tokens[:len(prompt_tokens)] == prompt_tokens
prompt_matches_rejected = rejected_tokens[:len(prompt_tokens)] == prompt_tokens

print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)
print(f"Prompt matches chosen prefix: {prompt_matches_chosen}")
print(f"Prompt matches rejected prefix: {prompt_matches_rejected}")

if not prompt_matches_chosen or not prompt_matches_rejected:
    print("\n⚠️  WARNING: Prompt tokens don't match! This will cause training issues.")

# Show the actual tool_calls in the text
print("\n" + "=" * 80)
print("TOOL_CALLS IN TEXT")
print("=" * 80)

if "<tool_call>" in full_chosen_text:
    start = full_chosen_text.find("<tool_call>")
    end = full_chosen_text.find("</tool_call>", start) + len("</tool_call>")
    print(f"\nChosen tool_call:")
    print(full_chosen_text[start:end])
else:
    print("\n⚠️  NO <tool_call> found in chosen text!")

if "<tool_call>" in full_rejected_text:
    start = full_rejected_text.find("<tool_call>")
    end = full_rejected_text.find("</tool_call>", start) + len("</tool_call>")
    print(f"\nRejected tool_call:")
    print(full_rejected_text[start:end])
else:
    print("\n⚠️  NO <tool_call> found in rejected text!")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Calculate the difference in completion tokens
completion_diff = abs(len(chosen_tokens) - len(prompt_tokens) - (len(rejected_tokens) - len(prompt_tokens)))
print(f"\nCompletion token difference: {completion_diff} tokens")

if completion_diff < 5:
    print("⚠️  WARNING: Chosen and rejected are almost identical in length!")
    print("   DPO may not learn effectively if the difference is too small.")
