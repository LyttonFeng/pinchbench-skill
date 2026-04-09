"""
PinchBench custom reward manager for veRL.

Online RL 版本：
  - 从 extra_info 中获取 trajectory 和 task_id
  - 调用 agent_loop/reward.py 计算 per-turn process reward
  - 在 <|im_end|> 位置赋 process reward
  - 最后一个 turn 叠加 terminal reward {-1, +1}

Reward 量级：
  - process reward: [-0.5, +0.3] per turn
  - terminal reward: {-1, +1}

三组 ablation (由 REWARD_MODE 环境变量控制)：
  - baseline: 纯 terminal reward
  - rule:     通用行为规则 + terminal
  - oracle:   规则 + 天眼 reference trajectory + terminal

使用方式（run_grpo_lora.sh 里）：
  reward.reward_manager.path=rl/train/reward_manager.py
  reward.reward_manager.name=PinchBenchRewardManager
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from verl.protocol import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Add agent_loop to path for reward computation
_AGENT_LOOP_DIR = str(Path(__file__).parent.parent / "agent_loop")
if _AGENT_LOOP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_LOOP_DIR)


def _find_im_end_positions(
    response_ids: torch.Tensor,
    im_end_token_id: int,
) -> list[int]:
    positions = []
    for pos, tid in enumerate(response_ids.tolist()):
        if tid == im_end_token_id:
            positions.append(pos)
    return positions


class PinchBenchRewardManager(AbstractRewardManager):
    """Per-turn process reward manager for PinchBench Online RL."""

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
        self.reward_mode = os.environ.get("REWARD_MODE", "self-judge")
        self.prm_vllm_base_url = os.environ.get("PRM_VLLM_BASE_URL", "http://localhost:8000/v1")
        self.prm_model = os.environ.get("PRM_MODEL", "Qwen3-4B")
        self.prm_api_key = os.environ.get("PRM_API_KEY", "dummy")

        im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.im_end_token_id = im_end_ids[-1] if im_end_ids else None

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        from reward import compute_episode_rewards

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attention_mask = data_item.batch["attention_mask"]

            valid_response_length = int(attention_mask[prompt_length:].sum().item())
            response_ids = data_item.batch["responses"][:valid_response_length]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except Exception:
                    extra_info = {}

            trajectory = extra_info.get("trajectory", [])
            task_id = extra_info.get("task_id", "")
            terminal_success = bool(extra_info.get("terminal_success", False))
            reward_mode = extra_info.get("reward_mode", self.reward_mode)

            terminal_reward = 1.0 if terminal_success else -1.0

            if not trajectory or self.im_end_token_id is None:
                reward_tensor[i, valid_response_length - 1] = terminal_reward
                reward_extra_info["terminal_reward"].append(terminal_reward)
                reward_extra_info["reward_mode"].append("fallback")
                continue

            # Compute per-turn rewards using self-judge (Qwen3-4B via vLLM)
            task_prompt = extra_info.get("task_prompt", "")
            per_turn_rewards = compute_episode_rewards(
                trajectory, terminal_success, task_id,
                mode=reward_mode,
                task_prompt=task_prompt,
                vllm_base_url=self.prm_vllm_base_url,
                judge_model=self.prm_model,
                judge_api_key=self.prm_api_key,
            )

            if not per_turn_rewards:
                reward_tensor[i, valid_response_length - 1] = terminal_reward
                reward_extra_info["terminal_reward"].append(terminal_reward)
                continue

            # Find <|im_end|> positions for reward assignment
            im_end_positions = _find_im_end_positions(response_ids, self.im_end_token_id)

            # Filter to only model-generated positions (if mask available)
            response_mask = data_item.batch.get("response_mask")
            if response_mask is not None:
                im_end_positions = [
                    pos for pos in im_end_positions
                    if pos < len(response_mask) and response_mask[pos] == 1
                ]

            n = min(len(per_turn_rewards), len(im_end_positions))
            for k in range(n):
                pos = im_end_positions[k]
                if pos < valid_response_length:
                    reward_tensor[i, pos] = per_turn_rewards[k]

            # Leftover rewards on last valid token
            if n < len(per_turn_rewards):
                leftover = sum(per_turn_rewards[n:])
                reward_tensor[i, valid_response_length - 1] += leftover

            reward_extra_info["terminal_reward"].append(terminal_reward)
            reward_extra_info["n_turns"].append(len(per_turn_rewards))
            reward_extra_info["per_turn_rewards"].append(per_turn_rewards[:n])
            reward_extra_info["reward_mode"].append(reward_mode)
            reward_extra_info["task_id"].append(task_id)
            reward_extra_info["total_process_reward"].append(
                sum(per_turn_rewards) - terminal_reward
            )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor
