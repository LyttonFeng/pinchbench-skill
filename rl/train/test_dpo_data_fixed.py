#!/usr/bin/env python3
"""Test that DPO data is balanced after fix."""

import json
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

data = Path("rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_pure_focused_train.jsonl").read_text().splitlines()

print("=" * 80)
print("DPO DATA BALANCE CHECK (After Bug 5 Fix)")
print("=" * 80)
print("\nSample | Chosen tokens | Rejected tokens | Ratio")
print("-------|---------------|-----------------|------")

total_chosen = 0
total_rejected = 0

for i, line in enumerate(data):
    row = json.loads(line)

    chosen_msgs = row["chosen"]["messages"]
    rejected_msgs = row["rejected"]["messages"]

    # Simulate fixed code: only first assistant turn
    chosen_completion = [chosen_msgs[1]]
    rejected_completion = [rejected_msgs[1]]

    chosen_text = tokenizer.apply_chat_template(chosen_completion, tokenize=False, add_generation_prompt=False)
    rejected_text = tokenizer.apply_chat_template(rejected_completion, tokenize=False, add_generation_prompt=False)

    chosen_tokens = len(tokenizer.encode(chosen_text, add_special_tokens=False))
    rejected_tokens = len(tokenizer.encode(rejected_text, add_special_tokens=False))

    total_chosen += chosen_tokens
    total_rejected += rejected_tokens

    ratio = rejected_tokens / chosen_tokens if chosen_tokens > 0 else 0

    print(f"{i:6d} | {chosen_tokens:13d} | {rejected_tokens:15d} | {ratio:5.1f}x")

avg_chosen = total_chosen / len(data)
avg_rejected = total_rejected / len(data)
avg_ratio = avg_rejected / avg_chosen

print("-------|---------------|-----------------|------")
print(f"{'AVG':>6s} | {avg_chosen:13.1f} | {avg_rejected:15.1f} | {avg_ratio:5.1f}x")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if avg_ratio < 2.0:
    print("✅ PASS: Average ratio < 2.0x (balanced)")
    print(f"   Chosen/Rejected are well-balanced ({avg_ratio:.1f}x)")
else:
    print("❌ FAIL: Average ratio >= 2.0x (unbalanced)")
    print(f"   Chosen/Rejected have length bias ({avg_ratio:.1f}x)")

print("\nComparison:")
print("  Before fix: 4-22x ratio (unbalanced, length bias)")
print(f"  After fix:  {avg_ratio:.1f}x ratio (should be ~1.0-2.0x)")
