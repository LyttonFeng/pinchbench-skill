"""
veRL custom reward function for PinchBench RL.

veRL 通过 custom_reward_function.path 加载这个文件，
调用 compute_reward(data_source, solution_str, ground_truth, extra_info) 计算 reward。

三层 reward（rl-algorithm.md §4）：
  1. immediate：幻觉/空回复 → -1
  2. next-state：tool 执行结果 → -0.5 / -0.1 / +0.2
  3. terminal：grading 分数 [0,1]，广播到所有 turn

veRL 接口：
  函数签名必须是 compute_score(solution_str, ground_truth, extra_info=None, **kwargs) -> float
  返回值是标量 reward，veRL 内部会广播到所有 response token。
"""

from __future__ import annotations

import json
import re
from typing import Any


# ---------- 配置 ----------
_R_NEXT_ERROR = -0.5
_R_NEXT_EMPTY = -0.1
_R_NEXT_OK = +0.2

_HALLUCINATION_PATTERNS = [
    r"I don't have access to",
    r"I cannot access",
    r"As an AI",
    r"I'm unable to",
    r"I don't have the ability",
]

_TOOL_ERROR_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"Error:",
    r"error:",
    r"Exception:",
    r"command not found",
    r"No such file or directory",
    r"Permission denied",
    r"ModuleNotFoundError",
    r"FileNotFoundError",
]


# ---------- 三层 reward ----------

def _immediate_reward(response: str, tool_calls: list) -> float:
    if not response.strip() and not tool_calls:
        return -1.0
    for pattern in _HALLUCINATION_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            return -1.0
    return 0.0


def _next_state_reward(tool_content: str) -> float:
    if not tool_content.strip():
        return _R_NEXT_EMPTY
    for pattern in _TOOL_ERROR_PATTERNS:
        if re.search(pattern, tool_content):
            return _R_NEXT_ERROR
    return _R_NEXT_OK


def _compute_turn_reward(
    trajectory: list[dict],
    terminal_reward: float,
) -> float:
    """
    计算整条 trajectory 的平均 turn reward。

    veRL 的 compute_score 返回一个标量，代表整个 response 的 reward。
    这里把所有 assistant turn 的 reward 取均值返回。
    """
    turn_rewards = []

    for i, turn in enumerate(trajectory):
        if turn.get("role") != "assistant":
            continue

        content = turn.get("content", "")
        tool_calls = turn.get("tool_calls", [])

        r_imm = _immediate_reward(content, tool_calls)
        if r_imm == -1.0:
            turn_rewards.append(-1.0)
            continue

        # 找紧跟的 tool turn
        r_next_list = []
        j = i + 1
        while j < len(trajectory) and trajectory[j].get("role") == "tool":
            r_next_list.append(_next_state_reward(trajectory[j].get("content", "")))
            j += 1

        r_next = sum(r_next_list) / len(r_next_list) if r_next_list else 0.0
        turn_rewards.append(r_next + terminal_reward)

    if not turn_rewards:
        return terminal_reward

    return sum(turn_rewards) / len(turn_rewards)


# ---------- veRL 接口 ----------

def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> float:
    """
    veRL custom reward function 入口。

    Args:
        solution_str: veRL rollout 生成的 response 文本
        ground_truth: reward_model.ground_truth（这里是 terminal_reward float）
        extra_info:   prepare_data.py 里透传的字段，包含完整 trajectory

    Returns:
        float: 该 response 的 reward
    """
    if extra_info is None:
        # fallback：直接用 terminal reward
        try:
            return float(ground_truth)
        except (TypeError, ValueError):
            return 0.0

    trajectory = extra_info.get("trajectory", [])
    terminal_reward = extra_info.get("terminal_reward", float(ground_truth or 0.0))

    if not trajectory:
        return terminal_reward

    return _compute_turn_reward(trajectory, terminal_reward)
