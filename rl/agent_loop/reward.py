"""
Process reward + terminal reward computation for PinchBench RL.

Three ablation modes:
  Mode A (baseline):  per-turn reward = 0, last turn gets terminal_reward
  Mode B (rule-only): per-turn reward from generic behavior rules
  Mode C (oracle):    Mode B + reference trajectory matching (open-eye judge)

Terminal reward: {-1, +1}  (task fail / succeed)
Process reward:  [-0.5, +0.3] per turn
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Generic behavior rules (Mode B) ──

def _extract_tool_calls(turn: dict) -> list[dict]:
    """Extract tool calls from an assistant turn."""
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


def _get_tool_name(turn: dict) -> Optional[str]:
    calls = _extract_tool_calls(turn)
    return calls[0]["name"] if calls else None


def _get_tool_args(turn: dict) -> dict:
    calls = _extract_tool_calls(turn)
    if not calls:
        return {}
    args = calls[0].get("arguments", {})
    if isinstance(args, str):
        try:
            import json
            return json.loads(args)
        except Exception:
            return {"_raw": args}
    return args


def _is_error_result(content: str) -> bool:
    error_patterns = [
        r"Traceback \(most recent call last\)",
        r"(?:Error|Exception):",
        r"command not found",
        r"No such file or directory",
        r"Permission denied",
        r"ModuleNotFoundError",
    ]
    for p in error_patterns:
        if re.search(p, content):
            return True
    return False


def generic_rule_reward(
    turn_index: int,
    turn: dict,
    prev_turns: list[dict],
    all_turns: list[dict],
    task_id: str,
) -> float:
    """Compute generic behavior rule reward for one assistant turn.

    Returns reward in [-0.5, +0.3].
    """
    content = turn.get("content", "")
    tool_name = _get_tool_name(turn)
    tool_args = _get_tool_args(turn)
    reward = 0.0

    # ── Positive signals ──

    # Used a tool (good — agent is taking action)
    if tool_name:
        reward += 0.05

    # Tool succeeded (next turn is tool result without error)
    next_turns = all_turns[turn_index + 1:]
    tool_results = []
    for t in next_turns:
        if t.get("role") == "tool":
            tool_results.append(t)
        else:
            break
    for tr in tool_results:
        tr_content = tr.get("content", "")
        if tr_content.strip() and not _is_error_result(tr_content):
            reward += 0.05

    # Read before write/edit (information gathering)
    if tool_name in ("write", "edit"):
        has_prior_read = any(
            _get_tool_name(t) == "read"
            for t in prev_turns
            if t.get("role") == "assistant"
        )
        if has_prior_read:
            reward += 0.05

    # Verification: read after write
    if tool_name == "read" and turn_index > 0:
        prev_assistant = [
            t for t in prev_turns if t.get("role") == "assistant"
        ]
        if prev_assistant:
            last_tool = _get_tool_name(prev_assistant[-1])
            if last_tool in ("write", "edit"):
                reward += 0.05

    # Web search with relevant query
    if tool_name == "web_search":
        query = tool_args.get("query", "")
        if query and len(query) > 5:
            reward += 0.03

    # ── Negative signals ──

    # Empty response without tool call
    if not content.strip() and not tool_name:
        reward -= 0.30

    # Hallucination patterns
    hallucination_patterns = [
        r"I don'?t have access to",
        r"I cannot access",
        r"As an AI",
        r"I'?m unable to",
    ]
    for p in hallucination_patterns:
        if re.search(p, content, re.IGNORECASE):
            reward -= 0.30
            break

    # Repeated failed command (same tool + same args + previous error)
    if tool_name and len(prev_turns) >= 2:
        prev_assistants = [
            t for t in prev_turns if t.get("role") == "assistant"
        ]
        if prev_assistants:
            prev_tool = _get_tool_name(prev_assistants[-1])
            prev_args = _get_tool_args(prev_assistants[-1])
            if prev_tool == tool_name and prev_args == tool_args:
                # Check if previous tool result was an error
                prev_idx = all_turns.index(prev_assistants[-1])
                for t in all_turns[prev_idx + 1:]:
                    if t.get("role") == "tool":
                        if _is_error_result(t.get("content", "")):
                            reward -= 0.30
                        break
                    elif t.get("role") == "assistant":
                        break

    # Tool error in result
    for tr in tool_results:
        if _is_error_result(tr.get("content", "")):
            reward -= 0.10

    return max(-0.5, min(0.3, reward))


# ── Oracle reward (Mode C): reference trajectory matching ──

# Reference trajectories encoded as structured step sequences
REFERENCE_TRAJECTORIES: dict[str, dict[str, Any]] = {
    "task_02_stock": {
        "min_turns": 3,
        "expected_tools": ["web_search", "write"],
        "key_milestones": [
            {"tool": "web_search", "args_pattern": r"stock|AAPL|price|Apple", "reward": 0.15},
            {"tool": "write", "args_pattern": r"stock_report", "reward": 0.25},
        ],
        "anti_patterns": [
            {"condition": "write_before_search", "penalty": -0.40},
        ],
        "content_quality": {
            "write": {"min_bytes": 200, "must_match": [r"\$?\d+\.?\d*", r"\d{4}"]},
        },
    },
    "task_12_skill_search": {
        "min_turns": 5,
        "expected_tools": ["read", "edit"],
        "key_milestones": [
            {"tool": "read", "args_pattern": r"config/", "reward": 0.15},
            {"tool": "edit", "args_pattern": r"config/", "reward": 0.15},
            {"tool": "read", "args_pattern": r"config/", "reward": 0.10, "after": "edit"},
        ],
        "anti_patterns": [
            {"condition": "sed_without_read", "penalty": -0.30},
            {"condition": "repeated_failure", "penalty": -0.40},
        ],
    },
    "task_10_workflow": {
        "min_turns": 3,
        "expected_tools": ["read", "write"],
        "key_milestones": [
            {"tool": "read", "args_pattern": r"config\.json", "reward": 0.15},
            {"tool": "write", "args_pattern": r"\.py$", "reward": 0.15},
            {"tool": "write", "args_pattern": r"NOTES|notes", "reward": 0.15},
        ],
        "content_quality": {
            "write_py": {"min_bytes": 500, "must_match": [r"import requests", r"import json"]},
            "write_notes": {"min_bytes": 500},
        },
    },
    "task_22_second_brain": {
        "min_turns": 4,
        "expected_tools": ["read", "write"],
        "key_milestones": [
            {"tool": "read", "args_pattern": r".", "reward": 0.15},
            {"tool": "write", "args_pattern": r".", "reward": 0.15},
        ],
        "anti_patterns": [
            {"condition": "write_before_read", "penalty": -0.20},
        ],
    },
    "task_16_email_triage": {
        "min_turns": 5,
        "expected_tools": ["read", "write"],
        "key_milestones": [
            {"tool": "read", "args_pattern": r"email|mail", "reward": 0.10, "repeatable": True},
            {"tool": "write", "args_pattern": r".", "reward": 0.15},
        ],
    },
    "task_19_spreadsheet_summary": {
        "min_turns": 4,
        "expected_tools": ["read", "exec", "write"],
        "key_milestones": [
            {"tool": "read", "args_pattern": r"\.(csv|xlsx)", "reward": 0.15},
            {"tool": "exec", "args_pattern": r"python|awk|pandas|cut", "reward": 0.15},
            {"tool": "write", "args_pattern": r"summary|report", "reward": 0.15},
        ],
        "anti_patterns": [
            {"condition": "write_without_exec_after_binary", "penalty": -0.40},
        ],
    },
    "task_18_market_research": {
        "min_turns": 4,
        "expected_tools": ["web_search", "write"],
        "key_milestones": [
            {"tool": "web_search", "args_pattern": r"market|APM|observability", "reward": 0.10},
            {"tool": "web_search", "args_pattern": r"vs|comparison|competitor", "reward": 0.10},
            {"tool": "write", "args_pattern": r"market_research", "reward": 0.15},
        ],
        "content_quality": {
            "write": {"min_bytes": 5000},
        },
        "anti_patterns": [
            {"condition": "single_search", "penalty": -0.20},
            {"condition": "outdated_year", "penalty": -0.20},
        ],
    },
    "task_24_polymarket_briefing": {
        "min_turns": 4,
        "expected_tools": ["web_search", "write"],
        "key_milestones": [
            {"tool": "web_search", "args_pattern": r"[Pp]olymarket", "reward": 0.15},
            {"tool": "web_search", "args_pattern": r".", "reward": 0.05, "repeatable": True},
            {"tool": "write", "args_pattern": r"polymarket_briefing", "reward": 0.15},
        ],
        "content_quality": {
            "write": {"min_bytes": 1200, "must_match": [r"## 1", r"## 2", r"## 3"]},
        },
        "anti_patterns": [
            {"condition": "outdated_year", "penalty": -0.20},
        ],
    },
}


def _check_milestone(
    turn: dict,
    milestone: dict,
    achieved_milestones: set[str],
    prev_turns: list[dict],
) -> float:
    """Check if this turn matches a reference milestone."""
    tool_name = _get_tool_name(turn)
    tool_args = _get_tool_args(turn)

    expected_tool = milestone["tool"]
    args_pattern = milestone.get("args_pattern", r".")

    if tool_name != expected_tool:
        return 0.0

    # Check args pattern against all string values in arguments
    args_str = " ".join(str(v) for v in tool_args.values()) if tool_args else ""
    if not re.search(args_pattern, args_str, re.IGNORECASE):
        return 0.0

    # Check ordering constraint
    if "after" in milestone:
        required_prior = milestone["after"]
        if required_prior not in achieved_milestones:
            return 0.0

    milestone_key = f"{expected_tool}:{args_pattern}"
    if not milestone.get("repeatable", False) and milestone_key in achieved_milestones:
        return 0.0

    achieved_milestones.add(milestone_key)
    return milestone.get("reward", 0.10)


def _check_anti_patterns(
    turn: dict,
    turn_index: int,
    all_turns: list[dict],
    task_ref: dict,
) -> float:
    """Check for anti-patterns in the trajectory up to this turn."""
    penalty = 0.0
    tool_name = _get_tool_name(turn)
    tool_args = _get_tool_args(turn)
    prev_assistants = [
        t for t in all_turns[:turn_index] if t.get("role") == "assistant"
    ]

    for ap in task_ref.get("anti_patterns", []):
        cond = ap["condition"]

        if cond == "write_before_search" and tool_name == "write":
            if not any(_get_tool_name(t) in ("web_search", "web_fetch") for t in prev_assistants):
                penalty += ap["penalty"]

        elif cond == "write_before_read" and tool_name == "write":
            if not any(_get_tool_name(t) == "read" for t in prev_assistants):
                penalty += ap["penalty"]

        elif cond == "sed_without_read" and tool_name == "exec":
            cmd = tool_args.get("command", tool_args.get("_raw", ""))
            if "sed" in str(cmd):
                if not any(_get_tool_name(t) == "read" for t in prev_assistants):
                    penalty += ap["penalty"]

        elif cond == "repeated_failure":
            # Already handled in generic rules
            pass

        elif cond == "single_search" and tool_name == "write":
            search_count = sum(
                1 for t in prev_assistants if _get_tool_name(t) == "web_search"
            )
            if search_count < 2:
                penalty += ap["penalty"]

        elif cond == "outdated_year" and tool_name == "web_search":
            query = str(tool_args.get("query", ""))
            if re.search(r"20(2[0-4]|1\d)", query):
                penalty += ap["penalty"]

        elif cond == "write_without_exec_after_binary" and tool_name == "write":
            saw_binary = False
            saw_exec = False
            for t in prev_assistants:
                if _get_tool_name(t) == "read":
                    idx = all_turns.index(t)
                    for tr in all_turns[idx + 1:]:
                        if tr.get("role") == "tool":
                            c = tr.get("content", "")
                            if "\x00" in c or len(c) > 500 and c.count("\\x") > 10:
                                saw_binary = True
                            break
                        elif tr.get("role") == "assistant":
                            break
                if _get_tool_name(t) == "exec":
                    saw_exec = True
            if saw_binary and not saw_exec:
                penalty += ap["penalty"]

    return penalty


def _check_content_quality(
    turn: dict,
    task_ref: dict,
) -> float:
    """Check content quality for write operations."""
    tool_name = _get_tool_name(turn)
    if tool_name != "write":
        return 0.0

    content = str(_get_tool_args(turn).get("content", ""))
    quality_checks = task_ref.get("content_quality", {})
    reward = 0.0

    for _key, spec in quality_checks.items():
        min_bytes = spec.get("min_bytes", 0)
        if len(content.encode("utf-8")) >= min_bytes:
            reward += 0.05
        else:
            reward -= 0.10

        for pattern in spec.get("must_match", []):
            if re.search(pattern, content):
                reward += 0.03

    return reward


def oracle_reward(
    turn_index: int,
    turn: dict,
    prev_turns: list[dict],
    all_turns: list[dict],
    task_id: str,
    achieved_milestones: set[str],
) -> float:
    """Compute oracle (open-eye) reward using reference trajectory.

    Only active when task_id has a reference trajectory.
    Returns additional reward on top of generic rules.
    """
    task_ref = REFERENCE_TRAJECTORIES.get(task_id)
    if task_ref is None:
        return 0.0

    reward = 0.0

    # Milestone matching
    for milestone in task_ref.get("key_milestones", []):
        reward += _check_milestone(turn, milestone, achieved_milestones, prev_turns)

    # Anti-pattern penalties
    reward += _check_anti_patterns(turn, turn_index, all_turns, task_ref)

    # Content quality
    reward += _check_content_quality(turn, task_ref)

    return max(-0.5, min(0.3, reward))


# ── Main reward computation ──

def compute_episode_rewards(
    trajectory: list[dict[str, Any]],
    terminal_success: bool,
    task_id: str,
    mode: str = "oracle",
) -> list[float]:
    """Compute per-assistant-turn rewards for a full episode.

    Args:
        trajectory: list of message dicts (all roles)
        terminal_success: whether PinchBench grading passed
        task_id: PinchBench task ID
        mode: "baseline" (A), "rule" (B), or "oracle" (C)

    Returns:
        List of rewards, one per assistant turn. Terminal reward is added
        to the last turn.
    """
    terminal_reward = 1.0 if terminal_success else -1.0

    assistant_indices = [
        i for i, t in enumerate(trajectory) if t.get("role") == "assistant"
    ]

    if not assistant_indices:
        return [terminal_reward]

    rewards = []
    achieved_milestones: set[str] = set()

    for seq_idx, turn_idx in enumerate(assistant_indices):
        turn = trajectory[turn_idx]
        prev_turns = trajectory[:turn_idx]

        if mode == "baseline":
            r = 0.0
        elif mode == "rule":
            r = generic_rule_reward(turn_idx, turn, prev_turns, trajectory, task_id)
        elif mode == "oracle":
            r_generic = generic_rule_reward(turn_idx, turn, prev_turns, trajectory, task_id)
            r_oracle = oracle_reward(
                turn_idx, turn, prev_turns, trajectory, task_id, achieved_milestones
            )
            r = r_generic + r_oracle
        else:
            r = 0.0

        # Clamp per-turn process reward
        r = max(-0.5, min(0.3, r))
        rewards.append(r)

    # Add terminal reward to last turn
    rewards[-1] += terminal_reward

    return rewards


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> float:
    """veRL-compatible reward function entry point.

    Called by veRL's reward pipeline. Returns a single scalar.
    For per-turn rewards, use compute_episode_rewards() directly.
    """
    if extra_info is None:
        extra_info = {}

    terminal_success = bool(extra_info.get("terminal_success", False))
    return 1.0 if terminal_success else -1.0
