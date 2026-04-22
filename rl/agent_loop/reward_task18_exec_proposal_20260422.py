"""
Task 18 hard-shaping reward proposal.

This file is intentionally NOT wired into the current training path.
It is a standalone proposal module so the existing reward implementation
remains untouched.

Intent:
  Force the policy boundary toward
    .xlsx -> exec + pandas/openpyxl
  instead of
    .xlsx -> read(binary garbage)

Design principles:
  - Do not require the first step to be correct.
  - Reward entering the correct parsing path at any later turn.
  - Make "late correction" still net positive, but worse than getting it
    right immediately.
"""

from __future__ import annotations

import json
from typing import Any, Optional


def _tool_args_text(tool_args: Any) -> str:
    if isinstance(tool_args, str):
        return tool_args
    try:
        return json.dumps(tool_args, ensure_ascii=False)
    except Exception:
        return str(tool_args)


def _extract_tool_calls(turn: dict[str, Any]) -> list[dict[str, Any]]:
    raw = turn.get("tool_calls", [])
    calls = []
    for tc in raw:
        if isinstance(tc, dict):
            func = tc.get("function", tc)
            calls.append({
                "name": func.get("name", tc.get("name", "")),
                "arguments": func.get("arguments", tc.get("arguments", {})),
            })
    return calls


def _get_tool_name(turn: dict[str, Any]) -> Optional[str]:
    calls = _extract_tool_calls(turn)
    return calls[0]["name"] if calls else None


def _get_tool_args(turn: dict[str, Any]) -> dict[str, Any]:
    calls = _extract_tool_calls(turn)
    if not calls:
        return {}
    args = calls[0].get("arguments", {})
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {"_raw": args}
    return args


def _match_tool_to_type(tool_name: str) -> str:
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


def _is_error_result(content: str) -> bool:
    error_markers = (
        "traceback",
        "error:",
        "exception:",
        "no such file or directory",
        "permission denied",
        "modulenotfounderror",
    )
    text = (content or "").lower()
    return any(marker in text for marker in error_markers)


def is_direct_xlsx_read(turn: dict[str, Any]) -> bool:
    tool_name = _get_tool_name(turn)
    tool_args = _get_tool_args(turn)
    if _match_tool_to_type(tool_name or "") != "read":
        return False
    return ".xlsx" in _tool_args_text(tool_args).lower()


def is_csv_read(turn: dict[str, Any]) -> bool:
    tool_name = _get_tool_name(turn)
    tool_args = _get_tool_args(turn)
    if _match_tool_to_type(tool_name or "") != "read":
        return False
    return ".csv" in _tool_args_text(tool_args).lower()


def is_exec_xlsx_parser(turn: dict[str, Any]) -> bool:
    tool_name = _get_tool_name(turn)
    tool_args = _get_tool_args(turn)
    if _match_tool_to_type(tool_name or "") != "exec":
        return False
    text = _tool_args_text(tool_args).lower()
    return (
        ".xlsx" in text
        and any(x in text for x in ("read_excel", "excelfile", "openpyxl", "load_workbook", "sheet_name"))
    )


def result_has_workbook_structure(tool_result: Optional[str]) -> bool:
    if not tool_result or _is_error_result(tool_result):
        return False
    text = tool_result.lower()
    markers = ("q1_expenses", "budgets", "sheet", "sheet_names", "columns", "employee", "department")
    return sum(1 for m in markers if m in text) >= 2


def task18_exec_event_reward(
    turn: dict[str, Any],
    prev_turns: list[dict[str, Any]],
    tool_result: Optional[str] = None,
) -> float:
    """
    Event-based shaping for task_18.

    Example behavior:
      - first turn read(xlsx): negative
      - later exec(read_excel): positive
      - if exec yields workbook structure: more positive
      - writing report after correct parser path: slightly positive
    """
    prev_assistant = [t for t in prev_turns if t.get("role") == "assistant"]
    seen_exec_parser = any(is_exec_xlsx_parser(t) for t in prev_assistant)
    seen_direct_xlsx_read = any(is_direct_xlsx_read(t) for t in prev_assistant)

    reward = 0.0
    tool_name = _get_tool_name(turn)
    tool_args = _get_tool_args(turn)
    args_text = _tool_args_text(tool_args).lower()

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

    if _match_tool_to_type(tool_name or "") == "write" and "data_summary.md" in args_text:
        reward += 0.10 if seen_exec_parser else -0.10

    return reward


def clip_task18_reward(reward: float) -> float:
    return max(-1.0, min(1.2, reward))

