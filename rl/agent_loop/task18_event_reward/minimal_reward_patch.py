"""
Minimal task-specific reward patch for task_18_spreadsheet_summary.

This module is intentionally standalone. It does not modify the current
reward path by itself.

How to use:
  - Import `apply_task18_minimal_patch_reward(...)` inside the current
    `compute_episode_rewards(_async)` loop after the base per-turn score
    is computed.
  - Keep terminal reward as-is.
  - Keep turn-level assignment as-is.
  - Optionally keep EMA normalization as-is for multi-task training.
"""

from __future__ import annotations

import json
from typing import Any, Optional


TASK18_ID = "task_18_spreadsheet_summary"


def _canonical_task_id(task_id: str) -> str:
    if task_id == "task_19_spreadsheet_summary":
        return TASK18_ID
    return task_id


def is_task18(task_id: str) -> bool:
    return _canonical_task_id(task_id) == TASK18_ID


def _extract_tool_calls(turn: dict[str, Any]) -> list[dict[str, Any]]:
    raw = turn.get("tool_calls", [])
    calls = []
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
    calls = _extract_tool_calls(turn)
    return calls[0]["name"] if calls else None


def get_tool_args(turn: dict[str, Any]) -> dict[str, Any]:
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


def render_args(tool_args: Any) -> str:
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
    return ".xlsx" in render_args(tool_args).lower()


def is_csv_read(turn: dict[str, Any]) -> bool:
    tool_name = get_tool_name(turn)
    tool_args = get_tool_args(turn)
    if match_tool_type(tool_name or "") != "read":
        return False
    return ".csv" in render_args(tool_args).lower()


def is_exec_xlsx_parser(turn: dict[str, Any]) -> bool:
    tool_name = get_tool_name(turn)
    tool_args = get_tool_args(turn)
    if match_tool_type(tool_name or "") != "exec":
        return False
    text = render_args(tool_args).lower()
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


def writes_report(turn: dict[str, Any]) -> bool:
    tool_name = get_tool_name(turn)
    if match_tool_type(tool_name or "") != "write":
        return False
    return "data_summary.md" in render_args(get_tool_args(turn)).lower()


def _count_previous(prev_turns: list[dict[str, Any]], pred) -> int:
    return sum(1 for t in prev_turns if t.get("role") == "assistant" and pred(t))


def apply_task18_minimal_patch_reward(
    task_id: str,
    turn: dict[str, Any],
    prev_turns: list[dict[str, Any]],
    tool_result: Optional[str] = None,
) -> float:
    """
    Minimal 8-event shaping layer.

    Returns additive reward delta for the current assistant turn.

    Event set:
      1. first direct read(.xlsx)                -> -0.8
      2. repeated direct read(.xlsx)             -> -0.4
      3. first exec+xlsx parser                  -> +1.0
      4. parser before any xlsx-read pollution   -> +0.3
      5. csv read before parser path             -> -0.1
      6. parser result shows workbook structure  -> +0.5
      7. parser exec failed                      -> -0.25
      8. write data_summary.md too early         -> -0.3
         write data_summary.md after parser path -> +0.1
    """
    if not is_task18(task_id):
        return 0.0

    prev_assistant = [t for t in prev_turns if t.get("role") == "assistant"]
    prev_xlsx_reads = _count_previous(prev_assistant, is_direct_xlsx_read)
    prev_exec_parsers = _count_previous(prev_assistant, is_exec_xlsx_parser)

    reward = 0.0

    if is_direct_xlsx_read(turn):
        reward += -0.8 if prev_xlsx_reads == 0 else -0.4

    if is_csv_read(turn) and prev_exec_parsers == 0:
        reward += -0.1

    if is_exec_xlsx_parser(turn):
        if prev_exec_parsers == 0:
            reward += 1.0
        if prev_xlsx_reads == 0:
            reward += 0.3
        if tool_result:
            if result_has_workbook_structure(tool_result):
                reward += 0.5
            elif is_error_result(tool_result):
                reward += -0.25

    if writes_report(turn):
        if prev_exec_parsers > 0 or is_exec_xlsx_parser(turn):
            reward += 0.1
        else:
            reward += -0.3

    return reward


def clip_task18_patched_turn_reward(base_turn_reward: float, patch_delta: float) -> float:
    """
    Clip patched per-turn reward into a broader range than the default [-0.5, 0.2].
    """
    reward = base_turn_reward + patch_delta
    return max(-1.0, min(1.5, reward))

