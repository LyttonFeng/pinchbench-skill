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
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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

    Returns a dict with 'score' and extra info for logging.
    veRL's NaiveRewardManager will place the score on the last valid token.
    """
    extra_info = extra_info or {}
    task_id = ground_truth or extra_info.get("task_id", "")
    trajectory = extra_info.get("trajectory", [])
    terminal_success = bool(extra_info.get("terminal_success", False))
    reward_mode = extra_info.get("reward_mode", os.environ.get("REWARD_MODE", "baseline"))
    task_prompt = extra_info.get("task_prompt", "")

    terminal_reward = 1.0 if terminal_success else -1.0

    if reward_mode == "baseline" or not trajectory:
        return {
            "score": terminal_reward,
            "terminal_reward": terminal_reward,
            "process_reward": 0.0,
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

        total_process = sum(per_turn_rewards)
        total_reward = total_process + terminal_reward

        return {
            "score": total_reward,
            "terminal_reward": terminal_reward,
            "process_reward": total_process,
            "reward_mode": reward_mode,
            "task_id": task_id,
            "n_turns": len(per_turn_rewards),
            "per_turn_rewards": per_turn_rewards,
        }
    except Exception as e:
        print(f"[PinchBench reward] Error computing process reward: {e}")
        return {
            "score": terminal_reward,
            "terminal_reward": terminal_reward,
            "process_reward": 0.0,
            "reward_mode": "error",
            "task_id": task_id,
            "error": str(e),
        }
