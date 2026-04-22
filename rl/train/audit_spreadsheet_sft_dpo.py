#!/usr/bin/env python3
"""Audit spreadsheet SFT/DPO data for first-step behavior quality.

Checks:
- Is the first assistant turn a tool call?
- Does the first assistant turn already use exec?
- Does it specifically use pandas/read_excel/openpyxl?
- Does DPO rejected start with continuation/session-status noise?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _first_assistant(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg
    return None


def _tool_call_text(msg: dict[str, Any]) -> str:
    tool_calls = msg.get("tool_calls") or []
    return json.dumps(tool_calls, ensure_ascii=False)


def audit_sft(path: Path) -> None:
    rows = _load_jsonl(path)
    total = len(rows)
    first_tool = 0
    first_exec = 0
    first_pandas = 0
    first_openpyxl = 0
    first_explore = 0

    for row in rows:
        msg = _first_assistant(row.get("messages", []))
        if not msg:
            continue
        tc_text = _tool_call_text(msg)
        if msg.get("tool_calls"):
            first_tool += 1
        if '"name": "exec"' in tc_text:
            first_exec += 1
        if "read_excel" in tc_text or "pandas" in tc_text or "pd." in tc_text:
            first_pandas += 1
        if "openpyxl" in tc_text:
            first_openpyxl += 1
        if "ls -la" in tc_text or "find . -type f" in tc_text:
            first_explore += 1

    print(f"\nSFT: {path}")
    print(f"rows={total}")
    print(f"first_assistant_has_tool={first_tool}/{total}")
    print(f"first_assistant_exec={first_exec}/{total}")
    print(f"first_assistant_pandas_or_read_excel={first_pandas}/{total}")
    print(f"first_assistant_openpyxl={first_openpyxl}/{total}")
    print(f"first_assistant_explore_ls_find={first_explore}/{total}")


def audit_dpo(path: Path) -> None:
    rows = _load_jsonl(path)
    total = len(rows)
    chosen_first_exec = 0
    chosen_first_pandas = 0
    rejected_first_read_xlsx = 0
    rejected_continue_prompt = 0
    rejected_session_status = 0
    prompt_mismatch = 0

    for row in rows:
        chosen_msgs = row.get("chosen", {}).get("messages", [])
        rejected_msgs = row.get("rejected", {}).get("messages", [])

        chosen_first = _first_assistant(chosen_msgs)
        rejected_first = _first_assistant(rejected_msgs)

        chosen_tc = _tool_call_text(chosen_first or {})
        rejected_tc = _tool_call_text(rejected_first or {})

        if '"name": "exec"' in chosen_tc:
            chosen_first_exec += 1
        if "read_excel" in chosen_tc or "pandas" in chosen_tc or "openpyxl" in chosen_tc:
            chosen_first_pandas += 1
        if '"name": "read"' in rejected_tc and "xlsx" in rejected_tc:
            rejected_first_read_xlsx += 1
        if rejected_msgs and rejected_msgs[0].get("role") == "user":
            if "Continue where you left off" in str(rejected_msgs[0].get("content", "")):
                rejected_continue_prompt += 1
        if '"name": "session_status"' in rejected_tc:
            rejected_session_status += 1
        if chosen_msgs and rejected_msgs:
            if chosen_msgs[0] != rejected_msgs[0]:
                prompt_mismatch += 1

    print(f"\nDPO: {path}")
    print(f"rows={total}")
    print(f"chosen_first_exec={chosen_first_exec}/{total}")
    print(f"chosen_first_pandas_openpyxl={chosen_first_pandas}/{total}")
    print(f"rejected_first_read_xlsx={rejected_first_read_xlsx}/{total}")
    print(f"rejected_continue_prompt={rejected_continue_prompt}/{total}")
    print(f"rejected_first_session_status={rejected_session_status}/{total}")
    print(f"chosen_rejected_prompt_mismatch={prompt_mismatch}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", action="append", default=[], help="SFT jsonl path; can pass multiple times")
    parser.add_argument("--dpo", action="append", default=[], help="DPO jsonl path; can pass multiple times")
    args = parser.parse_args()

    for p in args.sft:
        audit_sft(Path(p))
    for p in args.dpo:
        audit_dpo(Path(p))


if __name__ == "__main__":
    main()
