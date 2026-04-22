"""
Task-18-only reward implementation.

This module is intentionally standalone and does NOT modify the current
`rl/agent_loop/reward.py`.

Design:
  - No generic turn-level judge/rule reward.
  - Keep only:
      1. task_18 event-based shaping
      2. terminal reward

Intended use:
  - single-task RL on task_18_spreadsheet_summary
  - strongest test of whether task-specific shaping alone can force
    `.xlsx -> exec + pandas/openpyxl`
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


TASK18_ID = "task_18_spreadsheet_summary"
TASK19_ALIAS = "task_19_spreadsheet_summary"

TERMINAL_REWARD_WEIGHT = float(
    os.environ.get("PINCHBENCH_TERMINAL_REWARD_WEIGHT", "0.8")
)


def canonical_task_id(task_id: str) -> str:
    if task_id == TASK19_ALIAS:
        return TASK18_ID
    return task_id


def is_task18(task_id: str) -> bool:
    return canonical_task_id(task_id) == TASK18_ID


def extract_tool_calls(turn: dict[str, Any]) -> list[dict[str, Any]]:
    raw = turn.get("tool_calls", [])
    calls = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", tc)
        calls.append(
            {
                "name": func.get("name", tc.get("name", "")),
                "arguments": func.get("arguments", tc.get("arguments", {})),
            }
        )
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


def terminal_reward_raw(
    terminal_success: bool,
    task_id: str,
    workspace_path: str = "",
) -> float:
    if terminal_success:
        return 1.0
    if is_task18(task_id) and workspace_path:
        target = Path(workspace_path) / "data_summary.md"
        if not target.exists():
            return -1.0
    return 0.0


def task18_event_reward(
    turn: dict[str, Any],
    prev_turns: list[dict[str, Any]],
    tool_result: Optional[str] = None,
) -> float:
    """
    Minimal 8-event shaping layer.

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
    prev_assistant = [t for t in prev_turns if t.get("role") == "assistant"]
    prev_xlsx_reads = sum(1 for t in prev_assistant if is_direct_xlsx_read(t))
    prev_exec_parsers = sum(1 for t in prev_assistant if is_exec_xlsx_parser(t))

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


def clip_turn_reward(reward: float) -> float:
    return max(-1.0, min(1.5, reward))


def compute_episode_rewards(
    trajectory: list[dict[str, Any]],
    terminal_success: bool,
    task_id: str,
    workspace_path: str = "",
) -> list[float]:
    """
    Synchronous event-only reward.

    Returns one reward per assistant turn.
    No generic turn-level reward is used.
    """
    raw = terminal_reward_raw(terminal_success, task_id, workspace_path)
    terminal_reward = TERMINAL_REWARD_WEIGHT * raw

    assistant_indices = [
        i for i, t in enumerate(trajectory) if t.get("role") == "assistant"
    ]
    if not assistant_indices:
        return [terminal_reward]

    rewards: list[float] = []
    for turn_idx in assistant_indices:
        turn = trajectory[turn_idx]
        prev_turns = trajectory[:turn_idx]

        tool_result = None
        for t in trajectory[turn_idx + 1 :]:
            if t.get("role") == "tool":
                tool_result = t.get("content", "")
                break
            if t.get("role") == "assistant":
                break

        r = 0.0
        if is_task18(task_id):
            r += task18_event_reward(turn, prev_turns, tool_result)
        r = clip_turn_reward(r)
        rewards.append(r)

    rewards = [r + terminal_reward for r in rewards]
    return rewards


async def compute_episode_rewards_async(
    trajectory: list[dict[str, Any]],
    terminal_success: bool,
    task_id: str,
    task_prompt: str = "",
    mode: str = "task18-event-only",
    vllm_base_url: str = "",
    judge_model: str = "",
    judge_api_key: str = "",
    workspace_path: str = "",
) -> list[float]:
    """
    Async-compatible wrapper to match the existing reward module shape.

    Unused arguments are kept intentionally so this file can be wired in with
    minimal call-site changes.
    """
    return compute_episode_rewards(
        trajectory=trajectory,
        terminal_success=terminal_success,
        task_id=task_id,
        workspace_path=workspace_path,
    )
