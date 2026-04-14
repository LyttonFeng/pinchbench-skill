"""
PinchBench custom reward function for veRL.

veRL 0.7.1 custom_reward_function 接口：
  compute_score(data_source, solution_str, ground_truth, extra_info) -> float | dict

Online RL 场景下：
  - ground_truth = task_id (由 prepare_prompts.py 设置)
  - solution_str = 模型完整回复文本
  - extra_info 包含 task_id, trajectory, terminal_success 等

使用方式（run_reinforce_lora.sh 里）：
  reward.custom_reward_function.path=rl/train/reward_manager.py
  reward.custom_reward_function.name=compute_score

可选输出模式：
  - PINCHBENCH_REWARD_RETURN_MODE=scalar  → 使用 compute_score 的 episode scalar reward
  - PINCHBENCH_REWARD_RETURN_MODE=turn    → 使用 PinchBenchRewardManager 的 token reward tensor
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_AGENT_LOOP_DIR = str(Path(__file__).parent.parent / "agent_loop")
if _AGENT_LOOP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_LOOP_DIR)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Compute reward for a PinchBench episode.

    Returns a scalar score dict. veRL's NaiveRewardManager will place the score
    on the last valid token. Turn-level reward uses PinchBenchRewardManager.
    """
    extra_info = extra_info or {}
    task_id = ground_truth or extra_info.get("task_id", "")
    trajectory = extra_info.get("trajectory", [])
    terminal_success = bool(extra_info.get("terminal_success", False))
    reward_mode = extra_info.get("reward_mode", os.environ.get("REWARD_MODE", "baseline"))
    task_prompt = extra_info.get("task_prompt", "")

    terminal_reward_raw = 1.0 if terminal_success else -1.0
    terminal_reward_weight = float(
        os.environ.get("PINCHBENCH_TERMINAL_REWARD_WEIGHT", "0.3")
    )
    terminal_reward = terminal_reward_weight * terminal_reward_raw

    if reward_mode == "baseline" or not trajectory:
        return {
            "score": terminal_reward,
            "total_reward": terminal_reward,
            "terminal_reward": terminal_reward,
            "process_reward": 0.0,
            "terminal_success": terminal_success,
            "reward_mode": reward_mode if trajectory else "fallback",
            "task_id": task_id,
            "n_turns": 0,
        }

    try:
        from reward import compute_episode_rewards

        vllm_base_url = os.environ.get("PRM_VLLM_BASE_URL", "http://localhost:8000/v1")
        judge_model = os.environ.get("PRM_MODEL", "Qwen3-4B")
        judge_api_key = os.environ.get("PRM_API_KEY", "dummy")

        per_turn_rewards = compute_episode_rewards(
            trajectory, terminal_success, task_id,
            mode=reward_mode,
            task_prompt=task_prompt,
            vllm_base_url=vllm_base_url,
            judge_model=judge_model,
            judge_api_key=judge_api_key,
        )

        # compute_episode_rewards already adds terminal_reward to the last turn,
        # so the episode score is just the sum of per-turn rewards.
        total_reward = sum(per_turn_rewards)
        total_process = total_reward - terminal_reward

        return {
            "score": total_reward,
            "total_reward": total_reward,
            "terminal_reward": terminal_reward,
            "process_reward": total_process,
            "terminal_success": terminal_success,
            "reward_mode": reward_mode,
            "task_id": task_id,
            "n_turns": len(per_turn_rewards),
            "per_turn_rewards": per_turn_rewards,
        }
    except Exception as e:
        print(f"[PinchBench reward] Error computing process reward: {e}")
        return {
            "score": terminal_reward,
            "total_reward": terminal_reward,
            "terminal_reward": terminal_reward,
            "process_reward": 0.0,
            "terminal_success": terminal_success,
            "reward_mode": "error",
            "task_id": task_id,
            "error": str(e),
        }


def _field_at(non_tensor_batch: dict[str, Any], key: str, idx: int, default: Any = None) -> Any:
    """Read one item from veRL's non_tensor_batch across numpy/list/scalar layouts."""
    if key not in non_tensor_batch:
        return default
    value = non_tensor_batch[key]
    try:
        return value[idx]
    except Exception:
        return value


def _extra_info_at(non_tensor_batch: dict[str, Any], idx: int) -> dict[str, Any]:
    value = _field_at(non_tensor_batch, "extra_info", idx, {})
    extra = dict(value) if isinstance(value, dict) else {}
    # AgentLoopOutput.extra_fields may be flattened into non_tensor_batch by
    # some veRL versions instead of being nested under extra_info.
    for key in (
        "tool_rewards",
        "turn_scores",
        "total_reward",
        "reward_score",
        "process_reward",
        "terminal_reward",
        "terminal_success",
        "task_id",
    ):
        if key in non_tensor_batch and key not in extra:
            extra[key] = _field_at(non_tensor_batch, key, idx)
    return extra


def _valid_response_length(data_item: Any, response_size: int) -> int:
    """Infer the non-padding response length for a veRL data item."""
    batch = data_item.batch
    try:
        prompts = batch["prompts"]
        attention_mask = batch["attention_mask"]
        prompt_len = int(prompts.shape[-1])
        return int(attention_mask[prompt_len:].sum().item())
    except Exception:
        pass

    try:
        responses = batch["responses"]
        pad_token_id = getattr(data_item.meta_info, "pad_token_id", None)
        if pad_token_id is not None:
            return int((responses != pad_token_id).sum().item())
    except Exception:
        pass

    try:
        response_mask = batch["response_mask"]
        # Last fallback only: response_mask may exclude env/tool tokens, so it
        # undercounts multi-turn flattened responses.
        return int(response_mask.shape[-1])
    except Exception:
        return response_size


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


class PinchBenchRewardManager:
    """veRL reward manager that can consume token rewards from AgentLoopOutput.

    OpenClawAgentLoop stores per-token turn rewards in
    ``extra_fields["tool_rewards"]``. The default veRL custom reward function
    path collapses reward to a scalar, so this manager writes those token rewards
    directly into the reward tensor used by the trainer.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        num_examine: int = 0,
        compute_score: Any = None,
        **_: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score

    def __call__(self, data: Any, return_dict: bool = False) -> Any:
        import torch

        responses = data.batch["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)

        for i in range(len(data)):
            data_item = data[i]
            extra_info = _extra_info_at(data.non_tensor_batch, i)
            valid_len = _valid_response_length(data_item, responses.shape[-1])
            valid_len = max(0, min(valid_len, responses.shape[-1]))

            tool_rewards = _as_float_list(extra_info.get("tool_rewards"))
            if tool_rewards:
                n = min(valid_len, len(tool_rewards))
                if n > 0:
                    reward_tensor[i, :n] = torch.tensor(
                        tool_rewards[:n],
                        dtype=torch.float32,
                        device=reward_tensor.device,
                    )
                continue

            turn_scores = _as_float_list(extra_info.get("turn_scores"))
            if turn_scores:
                self._assign_turn_scores(
                    reward_tensor[i],
                    responses[i],
                    valid_len,
                    turn_scores,
                )
                continue

            # Fallback keeps the old scalar behavior if a rollout did not carry
            # turn-level rewards, e.g. an error path or an older agent loop.
            score = float(extra_info.get("total_reward", extra_info.get("reward_score", 0.0)))
            if valid_len > 0:
                reward_tensor[i, valid_len - 1] = score

        if return_dict:
            return {"reward_tensor": reward_tensor}
        return reward_tensor

    def _assign_turn_scores(
        self,
        row_reward: Any,
        row_response_ids: Any,
        valid_len: int,
        turn_scores: list[float],
    ) -> None:
        if valid_len <= 0:
            return

        im_end_id = None
        if self.tokenizer is not None:
            try:
                im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            except Exception:
                im_end_id = None

        if im_end_id is None:
            row_reward[valid_len - 1] += float(sum(turn_scores))
            return

        model_im_ends: list[int] = []
        for pos in range(valid_len):
            try:
                token_id = int(row_response_ids[pos].item())
            except Exception:
                token_id = int(row_response_ids[pos])
            if token_id == im_end_id:
                model_im_ends.append(pos)

        n = min(len(model_im_ends), len(turn_scores))
        for k in range(n):
            row_reward[model_im_ends[k]] = float(turn_scores[k])

        if n < len(turn_scores):
            row_reward[valid_len - 1] += float(sum(turn_scores[n:]))
