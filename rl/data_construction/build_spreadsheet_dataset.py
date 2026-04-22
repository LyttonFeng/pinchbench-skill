#!/usr/bin/env python3
"""Build synthetic spreadsheet task data for DPO collection and RL training.

This script materializes task_18_spreadsheet_summary variants into workspaces
plus JSONL manifests. The generated data is intentionally model-agnostic:

- RL consumes rl_{split}.jsonl with prompt, workspace files, expected answer,
  and reward spec.
- DPO collection consumes dpo_prompts_{split}.jsonl, then teacher/student
  rollouts can be paired by variant_id.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.data_construction.task_variants import get_generator


def _write_workspace(root: Path, files: dict[str, str | bytes]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for rel_path, content in sorted(files.items()):
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
            encoding = "binary"
            size = len(content)
            sha_preview = base64.b64encode(content[:24]).decode("ascii")
        else:
            path.write_text(content, encoding="utf-8")
            encoding = "utf-8"
            size = len(content.encode("utf-8"))
            sha_preview = content[:80]
        entries.append({
            "path": rel_path,
            "encoding": encoding,
            "bytes": size,
            "preview": sha_preview,
        })
    return entries


def _reward_spec() -> dict[str, Any]:
    return {
        "target_file": "data_summary.md",
        "max_turns_good": 8,
        "max_turns_hard": 10,
        "numeric_tolerance_pct": 1.0,
        "positive_checks": [
            "target_file_exists",
            "csv_total_revenue_correct",
            "csv_total_profit_correct",
            "csv_total_units_correct",
            "csv_top_region_correct",
            "csv_top_product_correct",
            "excel_total_q1_expenses_correct",
            "excel_top_department_correct",
            "excel_top_employee_correct",
            "budget_comparison_present",
        ],
        "negative_checks": [
            "missing_target_file",
            "claims_saved_without_file",
            "repeated_xlsx_binary_read",
            "turns_over_hard_limit",
            "empty_or_non_markdown_report",
        ],
    }


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    task_id = "task_18_spreadsheet_summary"
    generator = get_generator(task_id)
    reward_spec = _reward_spec()

    splits = [("train", args.train), ("val", args.val)]
    all_rows: list[dict[str, Any]] = []
    for split, count in splits:
        split_rows: list[dict[str, Any]] = []
        dpo_rows: list[dict[str, Any]] = []
        variants = generator.sample(n=count, seed=args.seed + (0 if split == "train" else 10_000))
        for idx, variant in enumerate(variants):
            variant_id = f"{task_id}-{split}-{idx:04d}-seed{variant.metadata['variant_seed']}"
            workspace_rel = Path("workspaces") / split / variant_id
            workspace_dir = out / workspace_rel
            workspace_files = _write_workspace(workspace_dir, variant.workspace_files)

            row = {
                "variant_id": variant_id,
                "task_id": task_id,
                "split": split,
                "prompt": variant.prompt,
                "workspace_dir": str(workspace_rel),
                "workspace_files": workspace_files,
                "expected": variant.expected,
                "reward_spec": reward_spec,
                "metadata": variant.metadata,
            }
            split_rows.append(row)
            all_rows.append(row)

            dpo_rows.append({
                "variant_id": variant_id,
                "task_id": task_id,
                "split": split,
                "prompt": variant.prompt,
                "workspace_dir": str(workspace_rel),
                "expected": variant.expected,
                "teacher_rollout_path": None,
                "student_rollout_path": None,
                "pair_status": "needs_rollouts",
                "acceptance": {
                    "chosen_min_score": args.chosen_min_score,
                    "rejected_max_score": args.rejected_max_score,
                    "max_chosen_turns": args.max_chosen_turns,
                },
            })

        _jsonl_write(out / f"rl_{split}.jsonl", split_rows)
        _jsonl_write(out / f"dpo_prompts_{split}.jsonl", dpo_rows)

    _jsonl_write(out / "manifest.jsonl", all_rows)
    summary = {
        "task_id": task_id,
        "train": args.train,
        "val": args.val,
        "seed": args.seed,
        "output_dir": str(out),
        "files": [
            "manifest.jsonl",
            "rl_train.jsonl",
            "rl_val.jsonl",
            "dpo_prompts_train.jsonl",
            "dpo_prompts_val.jsonl",
            "workspaces/",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="rl/data/generated/task_18_spreadsheet_summary")
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--val", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--chosen-min-score", type=float, default=0.9)
    parser.add_argument("--rejected-max-score", type=float, default=0.5)
    parser.add_argument("--max-chosen-turns", type=int, default=8)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
