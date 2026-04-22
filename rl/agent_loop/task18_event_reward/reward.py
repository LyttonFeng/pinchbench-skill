"""
Task 18 event-based reward proposal.

This file is standalone and does not replace the current
`rl/agent_loop/reward.py`.

Purpose:
  Provide a concrete reward design that strongly pushes
    .xlsx -> exec + pandas/openpyxl
  while still allowing late correction to earn positive credit.
"""

from __future__ import annotations

import json
from typing import Any, Optional


TASK_ID = "task_18_spreadsheet_summary"


def extract_tool_calls(turn: dict[str, Any]) -> list[dict[str, Any]]:
    raw = turn.get("tool_calls", [])
    calls: list[dict[str, Any]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", tc)
        calls.append({
            "name": func.get("name", tc.get("name", "")),
            "arguments": func.get("arguments", tc.get("arguments", {})),
        })
    return calls


def get_tool_name(turn: dict[str, Any]) -> Optional[str]:
    calls = extract_tool_calls(turn)
    return calls[0]["name"] if calls else None


def get_tool_args(turn: dict[str, Any]) -> dict[str, Any]:
    calls = extract_tool_calls(turn)
    if not calls:
        return {}
    args = calls[0].get("arguments", {})
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {"_raw": args}
    return args


def args_text(tool_args: Any) -> str:
    if isinstance(tool_args, str):
        return tool_args
    try:
        return json.dumps(tool_args, ensure_ascii=False)
    except Exception:
        return str(tool_args)


def match_tool_type(tool_name: str) -> str:
    name = tool_name.lower().replace("_", "")
    mapping = {
        "read": "read",
        "write": "write",
        "edit": "edit",
        "exec": "exec",
        "bash": "exec",
        "shell": "exec",
    }
    return mapping.get(name, tool_name.lower())


def is_error_result(content: str) -> bool:
    text = (content or "").lower()
    markers = (
        "traceback",
        "error:",
        "exception:",
        "no such file or directory",
        "permission denied",
        "modulenotfounderror",
    )
    return any(marker in text for marker in markers)


def is_direct_xlsx_read(turn: dict[str, Any]) -> bool:
    tool_name = get_tool_name(turn)
    tool_args = get_tool_args(turn)
    if match_tool_type(tool_name or "") != "read":
        return False
    return ".xlsx" in args_text(tool_args).lower()


def is_csv_read(turn: dict[str, Any]) -> bool:
    tool_name = get_tool_name(turn)
    tool_args = get_tool_args(turn)
    if match_tool_type(tool_name or "") != "read":
        return False
    return ".csv" in args_text(tool_args).lower()


def is_exec_xlsx_parser(turn: dict[str, Any]) -> bool:
    tool_name = get_tool_name(turn)
    tool_args = get_tool_args(turn)
    if match_tool_type(tool_name or "") != "exec":
        return False
    text = args_text(tool_args).lower()
    return (
        ".xlsx" in text
        and any(
            marker in text
            for marker in (
                "read_excel",
                "excelfile",
                "openpyxl",
                "load_workbook",
                "sheet_name",
            )
        )
    )


def result_has_workbook_structure(tool_result: Optional[str]) -> bool:
    if not tool_result or is_error_result(tool_result):
        return False
    text = tool_result.lower()
    markers = (
        "q1_expenses",
        "budgets",
        "sheet",
        "sheet_names",
        "columns",
        "employee",
        "department",
    )
    return sum(1 for m in markers if m in text) >= 2


def event_reward(
    turn: dict[str, Any],
    prev_turns: list[dict[str, Any]],
    tool_result: Optional[str] = None,
) -> float:
    """
    Event-based shaping for task_18.

    Logic:
      - Direct read(.xlsx) is strongly bad.
      - exec + pandas/openpyxl on .xlsx is strongly good.
      - Late correction is still rewarded.
      - Extracting workbook structure gets extra credit.
      - Writing data_summary.md after entering the correct parsing path gets
        a small bonus.
    """
    prev_assistant = [t for t in prev_turns if t.get("role") == "assistant"]
    seen_exec_parser = any(is_exec_xlsx_parser(t) for t in prev_assistant)
    seen_direct_xlsx_read = any(is_direct_xlsx_read(t) for t in prev_assistant)

    reward = 0.0
    tool_name = get_tool_name(turn)
    tool_args = get_tool_args(turn)
    rendered_args = args_text(tool_args).lower()

    if is_direct_xlsx_read(turn):
        reward -= 0.70 if not seen_direct_xlsx_read else 0.35

    if is_csv_read(turn) and not seen_exec_parser:
        reward -= 0.12

    if is_exec_xlsx_parser(turn):
        reward += 0.95 if not seen_exec_parser else 0.30
        if not seen_direct_xlsx_read:
            reward += 0.20

    if result_has_workbook_structure(tool_result):
        reward += 0.35

    if match_tool_type(tool_name or "") == "write" and "data_summary.md" in rendered_args:
        reward += 0.10 if seen_exec_parser else -0.10

    return reward


def clip_reward(reward: float) -> float:
    return max(-1.0, min(1.2, reward))

