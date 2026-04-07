"""
Reward 计算模块。

实现 rl-algorithm.md 中的三层 reward：
  - immediate：格式错误/幻觉 → -1，否则 0
  - terminal：grading 分数 [0,1]，广播到所有 turn
  - 合并：r_t = r_immediate + r_terminal（terminal 如果 r_immediate != -1）
"""

from __future__ import annotations

import re
from typing import Any

from schema import TurnMessage  # type: ignore


# immediate judge：检测低级错误
_HALLUCINATION_PATTERNS = [
    r"I don't have access to",
    r"I cannot access",
    r"As an AI",
    r"I'm unable to",
]


def immediate_reward(turn: TurnMessage) -> float:
    """
    Immediate judge：只看 response 本身，不看执行结果。

    返回：
      -1.0  低级错误（格式崩坏 / 明显幻觉 / 拒绝执行）
       0.0  格式正常，继续看 terminal reward
    """
    content = turn.content or ""

    # 检测幻觉/拒绝模式
    for pattern in _HALLUCINATION_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return -1.0

    # assistant 回复完全为空（没有文字也没有 tool call）
    if not content.strip() and not turn.tool_calls:
        return -1.0

    return 0.0


def compute_turn_rewards(
    trajectory: list[TurnMessage],
    terminal_reward: float,
) -> list[float]:
    """
    计算每个 assistant turn 的最终 reward。

    r_t = r_immediate + r_terminal（若 r_immediate != -1）
    r_t = -1（若 r_immediate == -1，直接惩罚，不加 terminal）

    返回长度与 trajectory 中 assistant turn 数量相同的 list。
    """
    rewards = []
    for turn in trajectory:
        if turn.role != "assistant":
            continue
        r_imm = immediate_reward(turn)
        if r_imm == -1.0:
            rewards.append(-1.0)
        else:
            rewards.append(terminal_reward)
    return rewards
