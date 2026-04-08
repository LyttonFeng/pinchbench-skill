"""
veRL custom reward function for PinchBench RL.

参考 OpenClaw-RL/toolcall-rl/generate_with_retool.py 的 process reward 实现：
  - rollout 时记录每个 assistant turn 的 token span（token_start, token_end）
  - 对每个 step 用规则打分（immediate + next-state），替代 PRM 模型
  - 把 per-step reward 按 token span 展开成 per-token reward tensor
  - terminal reward 叠加到所有 step

两种模式（通过 extra_info["reward_mode"] 控制）：

  A. outcome（对照组）：
     terminal reward 广播到所有 response token，不区分 step
     → extra_info["reward_mode"] = "outcome"

  B. process（实验组）：
     每个 assistant turn token 用 per-step reward（immediate + next-state + terminal）
     tool/observation token 的 reward = 0（不参与训练，loss_mask=0）
     → extra_info["reward_mode"] = "process"（默认）

veRL 接口：
  compute_score(solution_str, ground_truth, extra_info=None, **kwargs) -> float | list[float]
  返回 float → outcome reward（广播到所有 token）
  返回 list  → per-token reward（长度必须等于 response token 数）
"""

from __future__ import annotations

import re
from typing import Any

# ---------- reward 数值 ----------
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


# ---------- 规则 reward ----------

def _immediate_reward(content: str, tool_calls: list) -> float:
    if not content.strip() and not tool_calls:
        return -1.0
    for pattern in _HALLUCINATION_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return -1.0
    return 0.0


def _next_state_reward(tool_content: str) -> float:
    if not tool_content.strip():
        return _R_NEXT_EMPTY
    for pattern in _TOOL_ERROR_PATTERNS:
        if re.search(pattern, tool_content):
            return _R_NEXT_ERROR
    return _R_NEXT_OK


def _per_step_rewards(
    trajectory: list[dict],
    terminal_reward: float,
) -> list[dict]:
    """
    计算每个 assistant turn 的 step reward。

    返回：
      list of {
        "turn_index": int,     # 在 trajectory 中的位置
        "r_step": float,       # 该 step 的 reward
      }
    """
    step_rewards = []
    for i, turn in enumerate(trajectory):
        if turn.get("role") != "assistant":
            continue

        content = turn.get("content", "")
        tool_calls = turn.get("tool_calls", [])
        r_imm = _immediate_reward(content, tool_calls)

        if r_imm == -1.0:
            r_step = -1.0
        else:
            r_next_list = []
            j = i + 1
            while j < len(trajectory) and trajectory[j].get("role") == "tool":
                r_next_list.append(_next_state_reward(trajectory[j].get("content", "")))
                j += 1
            r_next = sum(r_next_list) / len(r_next_list) if r_next_list else 0.0
            r_step = r_next + terminal_reward

        step_rewards.append({"turn_index": i, "r_step": r_step})

    return step_rewards


# ---------- per-token reward 展开 ----------

def _build_token_reward(
    step_rewards: list[dict],
    step_token_spans: list[dict],
    total_response_tokens: int,
) -> list[float]:
    """
    把 per-step reward 按 token span 展开成 per-token reward tensor。

    参考 OpenClaw-RL step_scores_with_outcome 的做法：
      每个 assistant turn 对应的 token 位置赋该 step 的 reward
      其余位置（tool/observation token）赋 0（veRL 会用 loss_mask=0 屏蔽）

    Args:
        step_rewards: _per_step_rewards() 的输出
        step_token_spans: 每个 assistant turn 的 token 位置
          [{"turn_index": i, "token_start": int, "token_end": int}, ...]
        total_response_tokens: response 总 token 数

    Returns:
        list[float]，长度 = total_response_tokens
    """
    token_rewards = [0.0] * total_response_tokens

    # turn_index → r_step 映射
    reward_map = {sr["turn_index"]: sr["r_step"] for sr in step_rewards}

    for span in step_token_spans:
        turn_idx = span["turn_index"]
        start = span["token_start"]
        end = span["token_end"]
        r = reward_map.get(turn_idx, 0.0)
        for pos in range(start, min(end, total_response_tokens)):
            token_rewards[pos] = r

    return token_rewards


# ---------- veRL 接口 ----------

def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> float | list[float]:
    """
    veRL custom reward function 入口。

    extra_info 字段（由 prepare_data.py 透传）：
      - terminal_reward: float         grading 分数
      - trajectory: list[dict]         完整对话轮次
      - step_token_spans: list[dict]   每个 assistant turn 的 token 位置
        [{"turn_index": i, "token_start": int, "token_end": int}, ...]
      - total_response_tokens: int     response 总 token 数
      - reward_mode: str               "outcome" | "process"（默认 process）

    返回：
      float         outcome 模式，或 step_token_spans 缺失时的 fallback
      list[float]   process 模式，per-token reward tensor
    """
    if extra_info is None:
        try:
            return float(ground_truth)
        except (TypeError, ValueError):
            return 0.0

    terminal_reward = extra_info.get("terminal_reward", float(ground_truth or 0.0))
    trajectory = extra_info.get("trajectory", [])
    reward_mode = extra_info.get("reward_mode", "process")

    # outcome 模式：terminal reward 广播（对照组）
    if reward_mode == "outcome":
        return terminal_reward

    # process 模式：per-token reward（实验组）
    step_token_spans = extra_info.get("step_token_spans")
    total_response_tokens = extra_info.get("total_response_tokens")

    # step_token_spans 缺失时 fallback 到 outcome
    if not step_token_spans or not total_response_tokens:
        if not trajectory:
            return terminal_reward
        step_rewards = _per_step_rewards(trajectory, terminal_reward)
        if not step_rewards:
            return terminal_reward
        return sum(sr["r_step"] for sr in step_rewards) / len(step_rewards)

    step_rewards = _per_step_rewards(trajectory, terminal_reward)
    return _build_token_reward(step_rewards, step_token_spans, total_response_tokens)
