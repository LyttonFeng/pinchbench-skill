#!/usr/bin/env python3
"""Pair teacher/student spreadsheet rollouts into DPO preference data.

Expected rollout JSONL schema (one row per variant/model attempt):
{
  "variant_id": "...",
  "score": 1.0,
  "assistant_turns": 4,
  "transcript_path": "path/to/transcript.jsonl",
  "messages": [...]
}

If "messages" is omitted, the pair still records transcript_path so a later
converter can reconstruct the conversation from OpenClaw transcripts.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _best_teacher(rows: list[dict[str, Any]], min_score: float, max_turns: int) -> dict[str, Any] | None:
    candidates = [
        r for r in rows
        if float(r.get("score", 0.0)) >= min_score
        and int(r.get("assistant_turns", r.get("turns", 10**9))) <= max_turns
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda r: (-float(r.get("score", 0.0)), int(r.get("assistant_turns", r.get("turns", 10**9))),
                       str(r.get("transcript_path", ""))),
    )[0]


def _worst_student(rows: list[dict[str, Any]], max_score: float) -> dict[str, Any] | None:
    candidates = [r for r in rows if float(r.get("score", 1.0)) <= max_score]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda r: (float(r.get("score", 1.0)), -int(r.get("assistant_turns", r.get("turns", 0))),
                       str(r.get("transcript_path", ""))),
    )[0]


def build(args: argparse.Namespace) -> None:
    prompts = {r["variant_id"]: r for r in _load_jsonl(Path(args.prompts))}

    teachers_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(Path(args.teacher_rollouts)):
        teachers_by_variant[row["variant_id"]].append(row)

    students_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(Path(args.student_rollouts)):
        students_by_variant[row["variant_id"]].append(row)

    pairs = []
    skipped = []
    for variant_id, prompt_row in sorted(prompts.items()):
        chosen = _best_teacher(
            teachers_by_variant.get(variant_id, []),
            min_score=args.chosen_min_score,
            max_turns=args.max_chosen_turns,
        )
        rejected = _worst_student(
            students_by_variant.get(variant_id, []),
            max_score=args.rejected_max_score,
        )
        if not chosen or not rejected:
            skipped.append({
                "variant_id": variant_id,
                "has_chosen": bool(chosen),
                "has_rejected": bool(rejected),
                "teacher_attempts": len(teachers_by_variant.get(variant_id, [])),
                "student_attempts": len(students_by_variant.get(variant_id, [])),
            })
            continue

        pairs.append({
            "variant_id": variant_id,
            "task_id": prompt_row["task_id"],
            "prompt": prompt_row["prompt"],
            "workspace_dir": prompt_row["workspace_dir"],
            "expected": prompt_row.get("expected", {}),
            "chosen": {
                "score": chosen.get("score"),
                "assistant_turns": chosen.get("assistant_turns", chosen.get("turns")),
                "transcript_path": chosen.get("transcript_path"),
                "messages": chosen.get("messages"),
                "model": chosen.get("model"),
            },
            "rejected": {
                "score": rejected.get("score"),
                "assistant_turns": rejected.get("assistant_turns", rejected.get("turns")),
                "transcript_path": rejected.get("transcript_path"),
                "messages": rejected.get("messages"),
                "model": rejected.get("model"),
            },
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    skipped_path = output.with_suffix(".skipped.jsonl")
    with skipped_path.open("w", encoding="utf-8") as f:
        for row in skipped:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "pairs": len(pairs),
        "skipped": len(skipped),
        "output": str(output),
        "skipped_output": str(skipped_path),
        "chosen_min_score": args.chosen_min_score,
        "rejected_max_score": args.rejected_max_score,
        "max_chosen_turns": args.max_chosen_turns,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True, help="dpo_prompts_*.jsonl from build_spreadsheet_dataset.py")
    parser.add_argument("--teacher-rollouts", required=True)
    parser.add_argument("--student-rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chosen-min-score", type=float, default=0.9)
    parser.add_argument("--rejected-max-score", type=float, default=0.5)
    parser.add_argument("--max-chosen-turns", type=int, default=8)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
