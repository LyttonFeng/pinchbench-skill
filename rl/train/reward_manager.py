"""
PinchBench custom reward manager for veRL.

替换 veRL 默认的 naive reward manager，实现 step-level process reward：
  - 找到 response token 序列中每个 assistant turn 的最后一个 token（<|im_end|>）
  - 在该位置赋该 step 的 reward（immediate + next-state + terminal）
  - 其余位置 reward = 0

参考：
  - verl/workers/reward_manager/naive.py（接口结构）
  - OpenClaw-RL generate_with_retool.py（step_action_spans 思路）

使用方式（run_verl.sh 里）：
  reward.reward_manager.path=rl/train/reward_manager.py
  reward.reward_manager.name=PinchBenchRewardManager
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import torch

from verl.protocol import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager


# ---------- 规则 reward（和 reward_fn.py 保持一致）----------

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


def _immediate_reward(content: str, has_tool_call: bool) -> float:
    if not content.strip() and not has_tool_call:
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


def _compute_step_rewards(
    trajectory: list[dict],
    terminal_reward: float,
) -> list[float]:
    """
    计算每个 assistant turn 的 step reward，返回顺序与 assistant turn 在
    trajectory 中的顺序一致。
    """
    step_rewards = []
    for i, turn in enumerate(trajectory):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "")
        has_tool_call = bool(turn.get("tool_calls"))
        r_imm = _immediate_reward(content, has_tool_call)
        if r_imm == -1.0:
            step_rewards.append(-1.0)
            continue
        r_next_list = []
        j = i + 1
        while j < len(trajectory) and trajectory[j].get("role") == "tool":
            r_next_list.append(_next_state_reward(trajectory[j].get("content", "")))
            j += 1
        r_next = sum(r_next_list) / len(r_next_list) if r_next_list else 0.0
        step_rewards.append(r_next + terminal_reward)
    return step_rewards


# ---------- token 边界定位 ----------

def _find_im_end_positions(
    response_ids: torch.Tensor,
    im_end_token_id: int,
) -> list[int]:
    """
    在 response token 序列里找到所有 <|im_end|> 的位置。
    每个位置对应一个 turn 的结束。
    """
    positions = []
    for pos, tid in enumerate(response_ids.tolist()):
        if tid == im_end_token_id:
            positions.append(pos)
    return positions


# ---------- Custom Reward Manager ----------

class PinchBenchRewardManager(AbstractRewardManager):
    """
    Step-level process reward manager for PinchBench RL.

    对每个 assistant turn 的最后一个 token（<|im_end|>）赋该 step 的 reward：
      r_step = immediate + next_state + terminal

    其余 token 的 reward = 0。

    若 extra_info 里没有 trajectory（outcome 模式），退化为：
      最后一个 token 赋 terminal_reward。
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int = 1,
        compute_score=None,
        reward_fn_key: str = "data_source",
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        # 获取 <|im_end|> 的 token id
        im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.im_end_token_id = im_end_ids[-1] if im_end_ids else None

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attention_mask = data_item.batch["attention_mask"]

            valid_response_length = int(attention_mask[prompt_length:].sum().item())
            response_ids = data_item.batch["responses"][:valid_response_length]

            # extra_info 里拿 trajectory 和 terminal_reward
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            if isinstance(extra_info, str):
                import json
                try:
                    extra_info = json.loads(extra_info)
                except Exception:
                    extra_info = {}

            trajectory = extra_info.get("trajectory", [])
            terminal_reward = float(extra_info.get("terminal_reward", 0.0))
            reward_mode = extra_info.get("reward_mode", "process")

            if not trajectory or reward_mode == "outcome" or self.im_end_token_id is None:
                # fallback：outcome reward，放在最后一个 token
                reward_tensor[i, valid_response_length - 1] = terminal_reward
                reward_extra_info["terminal_reward"].append(terminal_reward)
                continue

            # 计算每个 assistant turn 的 step reward
            step_rewards = _compute_step_rewards(trajectory, terminal_reward)

            if not step_rewards:
                reward_tensor[i, valid_response_length - 1] = terminal_reward
                continue

            # 找 <|im_end|> 位置，每个位置对应一个 turn 结束
            im_end_positions = _find_im_end_positions(response_ids, self.im_end_token_id)

            # 只取 assistant turn 数量的位置（可能有 user/tool turn 的 <|im_end|>）
            # 按顺序取前 len(step_rewards) 个
            n = min(len(step_rewards), len(im_end_positions))
            for k in range(n):
                pos = im_end_positions[k]
                if pos < valid_response_length:
                    reward_tensor[i, pos] = step_rewards[k]

            # 如果 im_end 数量不够，把最后一个 step reward 放在最后一个 token
            if n < len(step_rewards):
                reward_tensor[i, valid_response_length - 1] += step_rewards[-1]

            reward_extra_info["terminal_reward"].append(terminal_reward)
            reward_extra_info["n_steps"].append(n)
            reward_extra_info["step_rewards"].append(step_rewards[:n])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor
