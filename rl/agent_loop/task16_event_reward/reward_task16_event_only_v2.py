from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


TERMINAL_REWARD_WEIGHT = float(
    os.environ.get("PINCHBENCH_TASK16_TERMINAL_REWARD_WEIGHT", "0.8")
)


def _lower(s: str | None) -> str:
    return (s or "").lower()


def _tool_calls(turn: dict[str, Any]) -> list[dict[str, Any]]:
    tc = turn.get("tool_calls")
    return tc if isinstance(tc, list) else []


def _tool_name(tc: dict[str, Any]) -> str:
    fn = tc.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name", ""))
    return str(tc.get("name", ""))


def _tool_args(tc: dict[str, Any]) -> str:
    fn = tc.get("function")
    if isinstance(fn, dict):
        args = fn.get("arguments", "")
    else:
        args = tc.get("arguments", "")
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False)
    except Exception:
        return str(args)


def _parse_args_json(args: str) -> dict[str, Any]:
    try:
        obj = json.loads(args)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def is_task16(task_id: str | None) -> bool:
    return task_id == "task_16_email_triage"


def _write_call(turn: dict[str, Any]) -> dict[str, Any] | None:
    for tc in _tool_calls(turn):
        if _tool_name(tc) != "write":
            continue
        args = _parse_args_json(_tool_args(tc))
        path = _lower(str(args.get("path", "")))
        if path.endswith("triage_report.md"):
            return args
    return None


def _writes_triage_report(turn: dict[str, Any]) -> bool:
    return _write_call(turn) is not None


def _read_calls(turn: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for tc in _tool_calls(turn):
        if _tool_name(tc) != "read":
            continue
        args = _parse_args_json(_tool_args(tc))
        path = str(args.get("path", ""))
        if path:
            out.append(path)
    return out


def _count_inbox_reads(turn: dict[str, Any]) -> int:
    return sum(1 for p in _read_calls(turn) if p.startswith("inbox/email_"))


def _reads_many_inbox_emails(turn: dict[str, Any]) -> bool:
    return _count_inbox_reads(turn) >= 8


def _bulk_read_count(prev_turns: list[dict[str, Any]]) -> int:
    return sum(1 for t in prev_turns if _reads_many_inbox_emails(t))


def _tool_result_text(tool_result: str | None) -> str:
    return _lower(tool_result)


def _context_overflow(tool_result: str | None) -> bool:
    txt = _tool_result_text(tool_result)
    return "maximum context length" in txt or ("32768" in txt and "input tokens" in txt)


def _structured_complete_report(turn: dict[str, Any]) -> bool:
    args = _write_call(turn)
    if not args:
        return False
    content = _lower(str(args.get("content", "")))
    has_incident_groups = "## incident groups" in content
    has_standalone_items = "## standalone items" in content
    has_priority = "priority" in content
    has_category = "category" in content
    has_action = "recommended action" in content or re.search(r"\baction\b", content) is not None
    p_count = len(re.findall(r"\bp[0-4]\b", content))
    flat_ok = bool(has_priority and has_category and has_action and p_count >= 8)
    grouped_ok = bool(
        has_incident_groups
        and has_standalone_items
        and has_priority
        and has_category
        and has_action
        and p_count >= 4
    )
    return flat_ok or grouped_ok


def _incident_group_schema_report(turn: dict[str, Any]) -> bool:
    args = _write_call(turn)
    if not args:
        return False
    content = _lower(str(args.get("content", "")))
    return "## incident groups" in content and "## standalone items" in content


def _generic_or_partial_report(turn: dict[str, Any]) -> bool:
    args = _write_call(turn)
    if not args:
        return False
    content = _lower(str(args.get("content", "")))
    if _structured_complete_report(turn):
        return False
    if len(re.findall(r"\bp[0-4]\b", content)) < 5:
        return True
    if "bigclient" not in content and "incident" not in content and "security" not in content:
        return True
    return False


def _mentions_incident_linkage(turn: dict[str, Any], tool_result: str | None) -> bool:
    text = "\n".join([_lower(turn.get("content", "")), _tool_result_text(tool_result)])
    return (
        ("alert" in text and "outage" in text)
        or ("latency" in text and "database incident" in text)
        or ("inc-20260217-001" in text)
    )


def _mentions_bigclient_high_priority(turn: dict[str, Any], tool_result: str | None) -> bool:
    text = "\n".join([_lower(turn.get("content", "")), _tool_result_text(tool_result)])
    return "bigclient" in text or "$2m annual contract" in text or "vp engineering" in text


def _mentions_security_high_priority(turn: dict[str, Any], tool_result: str | None) -> bool:
    text = "\n".join([_lower(turn.get("content", "")), _tool_result_text(tool_result)])
    return "security" in text or "password rotation" in text or "compliance" in text


def _early_switch_to_report(prev_turns: list[dict[str, Any]], turn: dict[str, Any]) -> bool:
    if not _writes_triage_report(turn):
        return False
    bulk_reads = _bulk_read_count(prev_turns)
    return bulk_reads <= 1


def _late_no_report(prev_turns: list[dict[str, Any]], turn: dict[str, Any]) -> bool:
    if _writes_triage_report(turn):
        return False
    bulk_reads = _bulk_read_count(prev_turns)
    current_bulk = _reads_many_inbox_emails(turn)
    return bulk_reads >= 1 and current_bulk


def _report_created_in_workspace(workspace_dir: str | None) -> bool:
    if not workspace_dir:
        return False
    return (Path(workspace_dir) / "triage_report.md").exists()


def terminal_reward_raw(terminal_success: bool, workspace_dir: str | None) -> float:
    if terminal_success:
        return 1.0
    if _report_created_in_workspace(workspace_dir):
        return -0.25
    return -1.0


def task16_event_reward(
    turn: dict[str, Any],
    prev_turns: list[dict[str, Any]],
    tool_result: str | None,
) -> float:
    r = 0.0

    if _writes_triage_report(turn):
        r += 0.2
    if _structured_complete_report(turn):
        r += 0.8
    if _incident_group_schema_report(turn):
        r += 0.2
    if _early_switch_to_report(prev_turns, turn):
        r += 0.5
    if _mentions_incident_linkage(turn, tool_result):
        r += 0.5
    if _mentions_bigclient_high_priority(turn, tool_result):
        r += 0.3
    if _mentions_security_high_priority(turn, tool_result):
        r += 0.3

    if _reads_many_inbox_emails(turn) and _bulk_read_count(prev_turns) >= 1:
        r -= 0.8
    if _late_no_report(prev_turns, turn):
        r -= 0.5
    if _generic_or_partial_report(turn):
        r -= 0.6
    if _context_overflow(tool_result):
        r -= 1.0

    return r


def _assistant_turn_indices(trajectory: list[dict[str, Any]]) -> list[int]:
    return [i for i, t in enumerate(trajectory) if t.get("role") == "assistant"]


def _clip_turn_reward(x: float) -> float:
    return max(-1.5, min(1.5, x))


def compute_episode_rewards(
    trajectory: list[dict[str, Any]],
    terminal_success: bool,
    task_id: str,
    workspace_path: str = "",
) -> list[float]:
    assistant_indices = _assistant_turn_indices(trajectory)
    terminal = TERMINAL_REWARD_WEIGHT * terminal_reward_raw(
        terminal_success=terminal_success,
        workspace_dir=workspace_path or None,
    )

    if not assistant_indices:
        return [terminal]

    per_turn: list[float] = []
    for pos, turn_idx in enumerate(assistant_indices):
        turn = trajectory[turn_idx]
        prev_turns = [trajectory[i] for i in assistant_indices[:pos]]

        tool_result = None
        for t in trajectory[turn_idx + 1 :]:
            if t.get("role") == "tool":
                tool_result = str(t.get("content", ""))
                break
            if t.get("role") == "assistant":
                break

        r = 0.0
        if is_task16(task_id):
            r += task16_event_reward(turn, prev_turns, tool_result)
        per_turn.append(_clip_turn_reward(r))

    return [x + terminal for x in per_turn]


async def compute_episode_rewards_async(
    trajectory: list[dict[str, Any]],
    terminal_success: bool,
    task_id: str,
    task_prompt: str = "",
    mode: str = "task16-event-only-v2",
    vllm_base_url: str = "",
    judge_model: str = "",
    judge_api_key: str = "",
    workspace_path: str = "",
) -> list[float]:
    return compute_episode_rewards(
        trajectory=trajectory,
        terminal_success=terminal_success,
        task_id=task_id,
        workspace_path=workspace_path,
    )
