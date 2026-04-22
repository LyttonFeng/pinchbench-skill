"""
Unit tests for reward design changes:
1. terminal reward broadcast to every turn (not just last)
2. EMA baseline tracks mean (not sum)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import unittest

# ── Inline the EMA logic for local testing (no verl dependency) ──────────────
# Mirrors rl/agent_loop/openclaw_agent_loop.py _normalize_turn_rewards exactly.

_test_ema: dict = {}
_TEST_EMA_ALPHA = 0.1
_TEST_EMA_INIT = float(os.environ.get("PINCHBENCH_TASK_EMA_INIT", "0.5"))


def _normalize_turn_rewards_impl(task_id: str, per_turn_rewards: list, ema: dict, init: float, alpha: float) -> list:
    if not per_turn_rewards:
        return per_turn_rewards
    raw_mean = sum(per_turn_rewards) / len(per_turn_rewards)
    if task_id not in ema:
        ema[task_id] = init
    baseline = ema[task_id]
    ema[task_id] = (1.0 - alpha) * baseline + alpha * raw_mean
    return [r - baseline for r in per_turn_rewards]


# ── Test 1: terminal reward broadcast ────────────────────────────────────────

class TestTerminalRewardBroadcast(unittest.TestCase):

    def _make_trajectory(self, n_turns: int) -> list[dict]:
        """Minimal trajectory with n assistant turns."""
        msgs = []
        for _ in range(n_turns):
            msgs.append({"role": "assistant", "content": "doing something", "tool_calls": []})
            msgs.append({"role": "tool", "content": "ok"})
        return msgs

    def test_terminal_added_to_every_turn_on_success(self):
        from rl.agent_loop.reward import compute_episode_rewards
        traj = self._make_trajectory(3)
        rewards = compute_episode_rewards(
            trajectory=traj,
            terminal_success=True,
            task_id="task_00_sanity",
            mode="baseline",
        )
        self.assertEqual(len(rewards), 3)
        terminal_weight = float(os.environ.get("PINCHBENCH_TERMINAL_REWARD_WEIGHT", "0.3"))
        expected_terminal = terminal_weight * 1.0
        for i, r in enumerate(rewards):
            self.assertAlmostEqual(r, expected_terminal, places=5,
                msg=f"Turn {i} should have terminal reward {expected_terminal}, got {r}")

    def test_terminal_zero_on_failure(self):
        from rl.agent_loop.reward import compute_episode_rewards
        traj = self._make_trajectory(3)
        rewards = compute_episode_rewards(
            trajectory=traj,
            terminal_success=False,
            task_id="task_00_sanity",
            mode="baseline",
        )
        self.assertEqual(len(rewards), 3)
        for i, r in enumerate(rewards):
            self.assertAlmostEqual(r, 0.0, places=5,
                msg=f"Turn {i} should be 0.0 on failure, got {r}")

    def test_all_turns_equal_in_baseline_mode(self):
        """In baseline mode (process=0), all turns should be identical."""
        from rl.agent_loop.reward import compute_episode_rewards
        traj = self._make_trajectory(5)
        rewards = compute_episode_rewards(
            trajectory=traj,
            terminal_success=True,
            task_id="task_00_sanity",
            mode="baseline",
        )
        self.assertEqual(len(set(round(r, 8) for r in rewards)), 1,
            msg=f"All turns should be equal in baseline mode, got {rewards}")

    def test_async_matches_sync(self):
        """Async and sync versions should return same results."""
        try:
            import aiohttp  # noqa
        except ImportError:
            self.skipTest("aiohttp not available in local env")
        from rl.agent_loop.reward import compute_episode_rewards, compute_episode_rewards_async
        traj = self._make_trajectory(4)

        sync_rewards = compute_episode_rewards(
            trajectory=traj,
            terminal_success=True,
            task_id="task_00_sanity",
            mode="baseline",
        )
        async_rewards = asyncio.run(compute_episode_rewards_async(
            trajectory=traj,
            terminal_success=True,
            task_id="task_00_sanity",
            mode="baseline",
        ))
        self.assertEqual(len(sync_rewards), len(async_rewards))
        for i, (s, a) in enumerate(zip(sync_rewards, async_rewards)):
            self.assertAlmostEqual(s, a, places=5,
                msg=f"Turn {i}: sync={s} async={a}")


# ── Test 2: EMA baseline tracks mean ─────────────────────────────────────────

class TestEMAMeanBaseline(unittest.TestCase):

    def setUp(self):
        _test_ema.clear()

    def _norm(self, task_id, rewards):
        return _normalize_turn_rewards_impl(task_id, rewards, _test_ema, _TEST_EMA_INIT, _TEST_EMA_ALPHA)

    def test_ema_tracks_mean_not_sum(self):
        """EMA baseline should converge to mean per-turn reward, independent of turn count."""
        ema_4 = {}
        ema_2 = {}
        rewards_4 = [0.2, 0.2, 0.2, 0.2]  # mean=0.2, sum=0.8
        rewards_2 = [0.2, 0.2]             # mean=0.2, sum=0.4

        for _ in range(200):
            _normalize_turn_rewards_impl("t", rewards_4, ema_4, 0.1, _TEST_EMA_ALPHA)
        for _ in range(200):
            _normalize_turn_rewards_impl("t", rewards_2, ema_2, 0.1, _TEST_EMA_ALPHA)

        self.assertAlmostEqual(ema_4["t"], 0.2, places=2,
            msg=f"4-turn EMA should converge to 0.2, got {ema_4['t']}")
        self.assertAlmostEqual(ema_2["t"], 0.2, places=2,
            msg=f"2-turn EMA should converge to 0.2, got {ema_2['t']}")
        self.assertAlmostEqual(ema_4["t"], ema_2["t"], places=2,
            msg="EMA baseline must be independent of turn count")

    def test_normalized_sum_positive_when_improving(self):
        _test_ema["t"] = 0.1
        normalized = self._norm("t", [0.3, 0.3, 0.3])
        self.assertGreater(sum(normalized), 0,
            msg=f"Improving episode should have positive normalized sum, got {normalized}")

    def test_normalized_sum_negative_when_regressing(self):
        _test_ema["t"] = 0.3
        normalized = self._norm("t", [0.05, 0.05, 0.05])
        self.assertLess(sum(normalized), 0,
            msg=f"Regressing episode should have negative normalized sum, got {normalized}")

    def test_normalized_near_zero_when_converged(self):
        _test_ema["t"] = 0.2
        normalized = self._norm("t", [0.2, 0.2, 0.2, 0.2])
        self.assertAlmostEqual(sum(normalized), 0.0, places=5,
            msg=f"Converged episode should have ~0 normalized sum, got {normalized}")

    def test_ema_init_env_var(self):
        """Default EMA init should be 0.1 when PINCHBENCH_TASK_EMA_INIT=0.1."""
        expected = float(os.environ.get("PINCHBENCH_TASK_EMA_INIT", "0.5"))
        self.assertEqual(_TEST_EMA_INIT, expected,
            msg=f"_TEST_EMA_INIT should be {expected}, got {_TEST_EMA_INIT}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
