"""
把 TrainingSample JSONL 转成 veRL 需要的 parquet 格式。

veRL 的数据格式要求（参考 examples/data_preprocess/gsm8k.py）：
  - prompt: list of {"role": ..., "content": ...}（chat messages）
  - reward_model: {"style": "rule", "ground_truth": ...}（或自定义 reward fn 用到的字段）
  - data_source: str（用于 reward fn 路由）
  - extra_info: dict（透传任意字段，reward_manager 里能拿到）

用法：
    python rl/train/prepare_data.py \
        --input rl/data/samples_rescored.jsonl \
        --output-dir rl/data/verl/ \
        --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import TrainingSample, TurnMessage


def _build_prompt(trajectory: list[TurnMessage]) -> list[dict]:
    """
    把 trajectory 前半部分（第一个 assistant turn 之前）作为 prompt。

    veRL 的 rollout 会从 prompt 末尾继续生成，所以 prompt = 用户消息。
    对于 multi-turn trajectory，把整条 trajectory 转成 messages list，
    最后一个 assistant turn 作为 response（veRL 会 re-generate 对比）。
    """
    messages = []
    for turn in trajectory:
        if turn.role == "user":
            messages.append({"role": "user", "content": turn.content})
        elif turn.role == "assistant":
            msg = {"role": "assistant", "content": turn.content}
            if turn.tool_calls:
                msg["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in turn.tool_calls
                ]
            messages.append(msg)
        elif turn.role == "tool":
            messages.append({
                "role": "tool",
                "content": turn.content,
                "name": turn.tool_name or "tool",
            })
    return messages


def sample_to_verl_row(sample: TrainingSample, reward_mode: str = "process") -> dict:
    """
    把一条 TrainingSample 转成 veRL parquet 的一行。

    veRL 训练时：
      - prompt = 完整对话（不含最后一个 assistant turn）
      - reward_fn 会拿到 extra_info 里的字段计算 reward
    """
    # prompt = 去掉最后一个 assistant turn 的 trajectory
    # （veRL 会重新 rollout 生成那个 assistant turn）
    last_assistant_idx = None
    for i in range(len(sample.trajectory) - 1, -1, -1):
        if sample.trajectory[i].role == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        prompt_turns = sample.trajectory
    else:
        prompt_turns = sample.trajectory[:last_assistant_idx]

    prompt = _build_prompt(prompt_turns)

    # extra_info 透传给 reward_manager
    extra_info = {
        "sample_id": sample.sample_id,
        "task_id": sample.task_id,
        "split": sample.split,
        "grading_type": sample.grading_type,
        "terminal_reward": sample.reward.terminal,
        "reward_breakdown": sample.reward.breakdown,
        "reward_mode": reward_mode,
        # 完整 trajectory 供 reward_manager 做 per-step 打分和 next-state 判断
        "trajectory": [t.to_dict() for t in sample.trajectory],
    }

    return {
        "data_source": "pinchbench",
        "prompt": prompt,
        "ability": "tool_use",
        "reward_model": {
            "style": "rule",
            "ground_truth": sample.reward.terminal,  # terminal reward 作为参考
        },
        "extra_info": extra_info,
    }


def convert(
    input_path: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    split_filter: str | None = "train",
    reward_mode: str = "process",
) -> None:
    """
    读取 JSONL，转成 veRL parquet，按 train/val split。
    """
    samples = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = TrainingSample.from_dict(json.loads(line))
                if not s.has_logprobs:
                    continue  # 没有 logprobs 的跳过
                if split_filter and s.split != split_filter:
                    continue
                samples.append(s)
            except Exception as e:
                print(f"跳过无效行: {e}")

    if not samples:
        print("没有可用样本，检查输入文件和过滤条件")
        sys.exit(1)

    print(f"加载 {len(samples)} 条样本，reward_mode={reward_mode}")

    rows = [sample_to_verl_row(s, reward_mode=reward_mode) for s in samples]

    # train/val split
    n_val = max(1, int(len(rows) * val_ratio))
    n_train = len(rows) - n_val
    train_rows = rows[:n_train]
    val_rows = rows[n_train:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    print(f"train: {len(train_rows)} 条 → {train_path}")
    print(f"val:   {len(val_rows)} 条 → {val_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TrainingSample JSONL → veRL parquet")
    parser.add_argument("--input", type=Path, required=True, help="rescored JSONL 文件")
    parser.add_argument("--output-dir", type=Path, default=Path("rl/data/verl"), help="输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例（默认 0.1）")
    parser.add_argument("--split", default="train", help="只处理指定 split（默认 train）")
    parser.add_argument("--reward-mode", default="process", choices=["outcome", "process"], help="reward 模式（默认 process）")
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        split_filter=args.split,
        reward_mode=args.reward_mode,
    )


if __name__ == "__main__":
    main()
