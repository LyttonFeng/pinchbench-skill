#!/usr/bin/env python3
"""Audit SFT samples and print how many assistant tokens are actually supervised."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from train_sft_lora_fixed import _build_labeled_example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [json.loads(line) for line in Path(args.data_path).read_text().splitlines() if line.strip()]

    counts = []
    for idx, row in enumerate(rows):
        ex = _build_labeled_example(
            tokenizer,
            row["messages"],
            max_length=args.max_length,
            tools=row.get("tools"),
        )
        supervised = sum(1 for x in ex["labels"] if x != -100)
        counts.append((idx, supervised, len(ex["input_ids"])))

    print(f"rows={len(counts)}")
    print(
        "min_supervised=", min(x[1] for x in counts) if counts else 0,
        "max_supervised=", max(x[1] for x in counts) if counts else 0,
        "avg_supervised=", round(sum(x[1] for x in counts) / len(counts), 1) if counts else 0.0,
        "zero_rows=", sum(1 for x in counts if x[1] == 0),
    )
    print("first_rows:")
    for idx, supervised, total in counts[: args.show]:
        print(f"  row={idx} supervised_tokens={supervised} total_tokens={total}")


if __name__ == "__main__":
    main()
