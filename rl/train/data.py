"""
训练数据加载模块。

从 TrainingSample JSONL 加载数据，过滤无效样本，
转换为 veRL 训练所需的格式。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from schema import TrainingSample  # type: ignore
from reward import compute_turn_rewards  # type: ignore

logger = logging.getLogger(__name__)


def load_samples(
    jsonl_path: Path,
    split: str | None = None,
    require_logprobs: bool = True,
    grading_types: list[str] | None = None,
) -> list[TrainingSample]:
    """
    加载 TrainingSample JSONL，过滤无效样本。

    Args:
        jsonl_path: JSONL 文件路径
        split: 只加载指定 split（train/val/test），None 表示全部
        require_logprobs: 是否过滤掉没有 logprobs 的样本
        grading_types: 只加载指定 grading_type，None 表示全部
    """
    samples = []
    skipped = 0

    with jsonl_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = TrainingSample.from_dict(json.loads(line))
            except Exception as exc:
                logger.warning("第 %d 行解析失败: %s", i, exc)
                skipped += 1
                continue

            # split 过滤
            if split and sample.split != split:
                skipped += 1
                continue

            # grading_type 过滤
            if grading_types and sample.grading_type not in grading_types:
                skipped += 1
                continue

            # timed_out 过滤
            if sample.timed_out:
                logger.debug("跳过超时样本: %s", sample.sample_id)
                skipped += 1
                continue

            # logprobs 过滤
            if require_logprobs and not sample.has_logprobs:
                logger.debug("跳过无 logprobs 样本: %s", sample.sample_id)
                skipped += 1
                continue

            # 必须有 assistant turn
            if not sample.assistant_turns:
                logger.debug("跳过无 assistant turn 样本: %s", sample.sample_id)
                skipped += 1
                continue

            samples.append(sample)

    logger.info(
        "加载 %d 条样本（跳过 %d）from %s",
        len(samples),
        skipped,
        jsonl_path,
    )
    return samples


def sample_to_verl(sample: TrainingSample) -> list[dict[str, Any]]:
    """
    把一条 TrainingSample 转成 veRL 训练所需的 step list。

    每个 assistant turn → 一个训练 step：
    {
        "prompt_tokens": [...],     # prompt 的 token ids（veRL 需要）
        "response_tokens": [...],   # response 的 token ids
        "logprobs_old": [...],      # re-score 得到的 old logprobs
        "reward": float,            # 该 turn 的 reward
        "sample_id": str,
        "task_id": str,
        "turn_index": int,
    }

    注意：token ids 需要在训练脚本里用 tokenizer 转换，
    这里只组织对话结构，不做 tokenization。
    """
    steps = []
    terminal_reward = sample.reward.terminal
    turn_rewards = compute_turn_rewards(sample.trajectory, terminal_reward)

    reward_idx = 0
    for i, turn in enumerate(sample.trajectory):
        if turn.role != "assistant":
            continue

        # 构建该 turn 的 prompt（之前所有消息）
        prompt_turns = sample.trajectory[:i]
        prompt_text = _turns_to_text(prompt_turns)
        response_text = turn.content

        # tool calls 序列化进 response
        if turn.tool_calls:
            tc_str = json.dumps(
                [{"name": tc.name, "arguments": tc.arguments} for tc in turn.tool_calls],
                ensure_ascii=False,
            )
            response_text = f"{response_text}\n[TOOL_CALLS]{tc_str}".strip()

        steps.append({
            "sample_id": sample.sample_id,
            "task_id": sample.task_id,
            "split": sample.split,
            "turn_index": i,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "logprobs_old": turn.logprobs or [],
            "reward": turn_rewards[reward_idx],
            "terminal_reward": terminal_reward,
            "grading_type": sample.grading_type,
        })
        reward_idx += 1

    return steps


def _turns_to_text(turns: list) -> str:
    """把对话轮次转成纯文本（用于 tokenization 前的中间格式）。"""
    import json as _json
    parts = []
    for turn in turns:
        role = turn.role
        content = turn.content or ""
        if role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            tc_str = ""
            if turn.tool_calls:
                tc_str = "\n[TOOL_CALLS]" + _json.dumps(
                    [{"name": tc.name, "arguments": tc.arguments} for tc in turn.tool_calls],
                    ensure_ascii=False,
                )
            parts.append(f"<|assistant|>\n{content}{tc_str}")
        elif role == "tool":
            name = turn.tool_name or "tool"
            parts.append(f"<|tool_result|>\n[{name}]: {content}")
    return "\n".join(parts)


import json  # noqa: E402
