"""
Reward 计算模块。

三层 reward（来自 rl-algorithm.md §4）：

  1. immediate_reward(x_t, y_t)
       只看 response 本身，不看执行结果。
       低级错误（幻觉/格式崩坏/拒绝执行）→ -1，否则 0。

  2. next_state_reward(y_t, z_{t+1})
       看 tool call 执行后的返回内容。
       执行出错 → -0.5，返回空 → -0.1，正常有内容 → +0.2。

  3. terminal_reward
       grading 函数输出 [0, 1]，广播到该 episode 所有 turn。

合并逻辑：
  if immediate == -1:
      r_t = -1.0                         # 低级错误，直接强惩罚
  else:
      r_t = next_state + terminal        # 执行效果 + 任务成败
"""

from __future__ import annotations

import re
from typing import Any

from schema import TurnMessage  # type: ignore


# ---------- Immediate reward ----------

_HALLUCINATION_PATTERNS = [
    r"I don't have access to",
    r"I cannot access",
    r"As an AI",
    r"I'm unable to",
    r"I don't have the ability",
]

_ERROR_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"Error:",
    r"error:",
    r"Exception:",
    r"command not found",
    r"No such file or directory",
    r"Permission denied",
    r"SyntaxError",
    r"ModuleNotFoundError",
    r"FileNotFoundError",
    r"ConnectionError",
    r"TimeoutError",
    r"stderr:",
]

# next-state reward 的具体数值
_R_NEXT_ERROR = -0.5    # tool 执行出错
_R_NEXT_EMPTY = -0.1    # tool 返回空
_R_NEXT_OK = +0.2       # tool 正常返回有内容


def immediate_reward(turn: TurnMessage) -> float:
    """
    Immediate judge：只看 response 本身（x_t, y_t），不看执行结果。

    返回：
      -1.0  低级错误（幻觉 / 拒绝执行 / 空回复）
       0.0  格式正常，继续看 next-state
    """
    content = turn.content or ""

    if not content.strip() and not turn.tool_calls:
        return -1.0

    for pattern in _HALLUCINATION_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return -1.0

    return 0.0


def next_state_reward(tool_turn: TurnMessage) -> float:
    """
    Next-state judge：看 tool call 执行后返回的 z_{t+1}。

    输入：role == "tool" 的 TurnMessage（tool 执行结果）。

    返回：
      _R_NEXT_ERROR  执行出错（stderr / exception / command not found）
      _R_NEXT_EMPTY  返回为空（tool 调用成功但没有实质内容）
      _R_NEXT_OK     正常返回，有实质内容
    """
    if tool_turn.role != "tool":
        return 0.0

    content = tool_turn.content or ""

    # 空返回
    if not content.strip():
        return _R_NEXT_EMPTY

    # 错误模式
    for pattern in _ERROR_PATTERNS:
        if re.search(pattern, content):
            return _R_NEXT_ERROR

    return _R_NEXT_OK


def compute_turn_rewards(
    trajectory: list[TurnMessage],
    terminal_reward: float,
) -> list[float]:
    """
    计算每个 assistant turn 的最终 reward。

    遍历 trajectory，对每个 assistant turn：
      1. 计算 immediate reward
      2. 找紧跟其后的 tool turn，计算 next-state reward
         （若 assistant turn 后没有 tool turn，next_state = 0）
      3. 合并：
           if immediate == -1 → r_t = -1
           else               → r_t = next_state + terminal

    返回长度与 assistant turn 数量相同的 list，顺序与 trajectory 一致。
    """
    rewards = []

    for i, turn in enumerate(trajectory):
        if turn.role != "assistant":
            continue

        r_imm = immediate_reward(turn)

        if r_imm == -1.0:
            rewards.append(-1.0)
            continue

        # 找紧跟其后的 tool turn（可能有多个连续 tool turn，取所有的均值）
        r_next_list = []
        j = i + 1
        while j < len(trajectory) and trajectory[j].role == "tool":
            r_next_list.append(next_state_reward(trajectory[j]))
            j += 1

        r_next = sum(r_next_list) / len(r_next_list) if r_next_list else 0.0

        rewards.append(r_next + terminal_reward)

    return rewards


def reward_stats(rewards: list[float]) -> dict[str, float]:
    """计算 reward 统计量，用于 logging 和诊断。"""
    if not rewards:
        return {}
    return {
        "mean": sum(rewards) / len(rewards),
        "min": min(rewards),
        "max": max(rewards),
        "n": float(len(rewards)),
        "n_immediate_penalty": float(sum(1 for r in rewards if r == -1.0)),
    }
