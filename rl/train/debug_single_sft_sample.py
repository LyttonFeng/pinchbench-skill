#!/usr/bin/env python3
"""Debug supervision coverage for a single SFT sample.

This script is for answering one narrow question:
are the crucial tool-calling tokens actually supervised?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def _load_row(path: Path, index: int) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[index]


def _build_labeled_example(
    tokenizer,
    messages: list[dict],
    max_length: int,
    tools: list[dict] | None = None,
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

    return {
        "input_ids": input_ids[:max_length],
        "labels": labels[:max_length],
        "attention_mask": [1] * min(len(input_ids), max_length),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    row = _load_row(Path(args.data_path), args.row)
    messages = row["messages"]
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=row.get("tools"),
        tokenize=False,
        add_generation_prompt=False,
    )
    ex = _build_labeled_example(
        tokenizer,
        messages,
        max_length=args.max_length,
        tools=row.get("tools"),
    )

    input_ids = ex["input_ids"]
    labels = ex["labels"]
    supervised_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    print("=== SAMPLE ===")
    print(json.dumps(row, ensure_ascii=False, indent=2)[:8000])
    print("\n=== RENDERED TEMPLATE TEXT ===")
    print(rendered[:8000])
    print("\n=== SUPERVISED TOKEN STATS ===")
    print({
        "total_tokens": len(input_ids),
        "supervised_tokens": len(supervised_ids),
        "coverage": round(len(supervised_ids) / max(1, len(input_ids)), 4),
    })
    print("\n=== SUPERVISED TEXT ===")
    print(supervised_text[:8000])

    keywords = ["exec", "read", "pandas", "read_excel", "company_expenses.xlsx", "quarterly_sales.csv"]
    print("\n=== KEYWORD COVERAGE ===")
    for kw in keywords:
        print(
            json.dumps(
                {
                    "keyword": kw,
                    "in_rendered": kw in rendered,
                    "in_full_decoded": kw in full_text,
                    "in_supervised_text": kw in supervised_text,
                    "rendered_count": rendered.count(kw),
                    "supervised_count": supervised_text.count(kw),
                },
                ensure_ascii=False,
            )
        )

    print("\n=== SUPERVISED TOKEN PIECES (first 120) ===")
    pieces = []
    for tid, lab in zip(input_ids, labels):
        if lab != -100:
            pieces.append(tokenizer.decode([tid], skip_special_tokens=False))
        if len(pieces) >= 120:
            break
    for i, piece in enumerate(pieces, 1):
        print(f"{i:03d}: {piece!r}")


if __name__ == "__main__":
    main()
