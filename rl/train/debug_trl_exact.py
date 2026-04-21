#!/usr/bin/env python3
"""Reproduce TRL's exact tokenization logic to debug mismatch."""

import json
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load one sample
data_path = Path("rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_pure_focused_train.jsonl")
row = json.loads(data_path.read_text().splitlines()[0])

chosen_msgs = row["chosen"]["messages"]
rejected_msgs = row["rejected"]["messages"]

# Extract prompt (first user message)
prompt_msgs = [chosen_msgs[0]]

print("=" * 80)
print("REPRODUCING TRL TOKENIZATION")
print("=" * 80)

# TRL's logic (from DPOTrainer._tokenize):
# 1. Apply chat template to prompt
# 2. Apply chat template to prompt + chosen
# 3. Check if prompt tokens match the start of (prompt + chosen) tokens

print("\n1. Tokenize prompt messages:")
print(f"   Messages: {len(prompt_msgs)}")
prompt_text = tokenizer.apply_chat_template(
    prompt_msgs,
    tokenize=False,
    add_generation_prompt=False
)
print(f"   Text length: {len(prompt_text)}")
print(f"   Text (last 100): ...{prompt_text[-100:]}")

# Tokenize with different parameters to see which one TRL uses
print("\n2. Try different tokenization parameters:")

# Option A: No special tokens
prompt_tokens_a = tokenizer.encode(prompt_text, add_special_tokens=False)
print(f"   A (add_special_tokens=False): {len(prompt_tokens_a)} tokens")

# Option B: With special tokens
prompt_tokens_b = tokenizer.encode(prompt_text, add_special_tokens=True)
print(f"   B (add_special_tokens=True): {len(prompt_tokens_b)} tokens")

# Option C: Using tokenizer() directly
prompt_tokens_c = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
print(f"   C (tokenizer() no special): {len(prompt_tokens_c)} tokens")

# Option D: Using tokenizer() with special tokens
prompt_tokens_d = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
print(f"   D (tokenizer() with special): {len(prompt_tokens_d)} tokens")

print("\n3. Tokenize full chosen conversation:")
full_chosen_text = tokenizer.apply_chat_template(
    chosen_msgs,
    tokenize=False,
    add_generation_prompt=False
)
print(f"   Text length: {len(full_chosen_text)}")

# Try same options
chosen_tokens_a = tokenizer.encode(full_chosen_text, add_special_tokens=False)
chosen_tokens_b = tokenizer.encode(full_chosen_text, add_special_tokens=True)
chosen_tokens_c = tokenizer(full_chosen_text, add_special_tokens=False)["input_ids"]
chosen_tokens_d = tokenizer(full_chosen_text, add_special_tokens=True)["input_ids"]

print(f"   A (no special): {len(chosen_tokens_a)} tokens")
print(f"   B (with special): {len(chosen_tokens_b)} tokens")
print(f"   C (tokenizer() no special): {len(chosen_tokens_c)} tokens")
print(f"   D (tokenizer() with special): {len(chosen_tokens_d)} tokens")

print("\n4. Check matches for each option:")
for opt, (p_tok, c_tok) in [
    ("A", (prompt_tokens_a, chosen_tokens_a)),
    ("B", (prompt_tokens_b, chosen_tokens_b)),
    ("C", (prompt_tokens_c, chosen_tokens_c)),
    ("D", (prompt_tokens_d, chosen_tokens_d)),
]:
    matches = c_tok[:len(p_tok)] == p_tok
    print(f"   Option {opt}: {'✅ MATCH' if matches else '❌ MISMATCH'}")

    if not matches:
        # Find first difference
        for i in range(min(len(p_tok), len(c_tok))):
            if p_tok[i] != c_tok[i]:
                print(f"      First diff at pos {i}: prompt={p_tok[i]} vs chosen={c_tok[i]}")
                print(f"      Prompt token: '{tokenizer.decode([p_tok[i]])}'")
                print(f"      Chosen token: '{tokenizer.decode([c_tok[i]])}'")
                break

print("\n5. Check if TRL uses apply_chat_template with tokenize=True:")
# This is what TRL might actually do
prompt_result = tokenizer.apply_chat_template(
    prompt_msgs,
    tokenize=True,
    add_generation_prompt=False
)
chosen_result = tokenizer.apply_chat_template(
    chosen_msgs,
    tokenize=True,
    add_generation_prompt=False
)

print(f"   Prompt type: {type(prompt_result)}")
print(f"   Chosen type: {type(chosen_result)}")

# Extract token IDs from BatchEncoding
if hasattr(prompt_result, 'input_ids'):
    prompt_tokens_direct = prompt_result.input_ids
    chosen_tokens_direct = chosen_result.input_ids
    print(f"   Extracted input_ids from BatchEncoding")
else:
    prompt_tokens_direct = prompt_result
    chosen_tokens_direct = chosen_result

print(f"   Prompt (direct): {len(prompt_tokens_direct)} tokens")
print(f"   Chosen (direct): {len(chosen_tokens_direct)} tokens")

matches_direct = chosen_tokens_direct[:len(prompt_tokens_direct)] == prompt_tokens_direct
print(f"   Direct match: {'✅ MATCH' if matches_direct else '❌ MISMATCH'}")

if not matches_direct:
    print(f"\n   First 10 prompt tokens: {prompt_tokens_direct[:10]}")
    print(f"   First 10 chosen tokens: {chosen_tokens_direct[:10]}")

    for i in range(min(len(prompt_tokens_direct), len(chosen_tokens_direct))):
        if prompt_tokens_direct[i] != chosen_tokens_direct[i]:
            print(f"\n   First diff at pos {i}:")
            print(f"      Prompt: {prompt_tokens_direct[i]} = '{tokenizer.decode([prompt_tokens_direct[i]])}'")
            print(f"      Chosen: {chosen_tokens_direct[i]} = '{tokenizer.decode([chosen_tokens_direct[i]])}'")
            break
