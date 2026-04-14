# PinchBench RL configs

This directory stores small, human-readable snapshots of the reward / loss
setup used in training experiments.

## Files

- `reward_scalar_current.yaml`
  - Current online RL setup.
  - Scalar episode reward.
  - Matches the existing `rl/train/run_reinforce_lora.sh` default behavior.

- `reward_turn_level.yaml`
  - Turn-level reward setup.
  - Intended for experiments where different assistant turns should carry
    different reward signals and drive a turn-level loss.
  - Uses `PinchBenchRewardManager`, which consumes `extra_info.tool_rewards`
    from `OpenClawAgentLoop` and writes them into veRL's reward tensor.

## Runtime switch

The main training script now accepts:

- `PINCHBENCH_REWARD_RETURN_MODE=scalar`
- `PINCHBENCH_REWARD_RETURN_MODE=turn`

The default remains `scalar` so the current training path is unchanged.

In `turn` mode, `rl/train/run_reinforce_lora.sh` switches from
`reward.custom_reward_function` to `reward.reward_manager.name=PinchBenchRewardManager`.
