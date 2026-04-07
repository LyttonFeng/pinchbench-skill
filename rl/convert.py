"""
openclaw transcript (JSONL) → TrainingSample

用法：
    from rl.convert import transcript_to_sample
    sample = transcript_to_sample(
        transcript_path=Path("results/0004_transcripts/task_01_calendar.jsonl"),
        task_id="task_01_calendar",
        prompt="Schedule a meeting...",
        grading_type="automated",
        reward_terminal=1.0,
        reward_breakdown={"file_created": 1.0, ...},
        model_id="qwen-plus",
        seed=0,
        run_index=0,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from schema import GRPOGroup, Reward, TrainingSample, TurnMessage, UsageStats, split_for_seed


def _parse_openclaw_transcript(path: Path) -> list[dict[str, Any]]:
    """读取 openclaw JSONL transcript，返回事件列表。"""
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _events_to_turns(events: list[dict[str, Any]]) -> list[TurnMessage]:
    """将 openclaw 事件流转换为 TurnMessage 列表。

    openclaw transcript 格式：
      type=message, message.role=user      → TurnMessage(role="user")
      type=message, message.role=assistant → TurnMessage(role="assistant", tool_calls=[...])
      type=message, message.role=toolResult → TurnMessage(role="tool")
    """
    turns: list[TurnMessage] = []

    for ev in events:
        if ev.get("type") != "message":
            continue

        msg = ev.get("message", {})
        role = msg.get("role", "")
        content_items = msg.get("content", [])

        if role == "user":
            text = ""
            for item in content_items:
                if item.get("type") == "text":
                    text += item.get("text", "")
            if text.strip():
                turns.append(TurnMessage(role="user", content=text.strip()))

        elif role == "assistant":
            text = ""
            tool_calls = []
            for item in content_items:
                if item.get("type") == "text":
                    text += item.get("text", "")
                elif item.get("type") == "toolCall":
                    tool_calls.append({
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", {}),
                    })
            from schema import ToolCall
            turns.append(TurnMessage(
                role="assistant",
                content=text.strip(),
                tool_calls=[
                    ToolCall(name=tc["name"], arguments=tc["arguments"])
                    for tc in tool_calls
                ],
                logprobs=None,  # openclaw 不暴露 logprobs，vLLM 采样时填入
            ))

        elif role == "toolResult":
            tool_name = msg.get("toolName", "")
            text = ""
            for item in content_items:
                if item.get("type") == "text":
                    text += item.get("text", "")
            turns.append(TurnMessage(
                role="tool",
                content=text.strip(),
                tool_name=tool_name,
            ))

    return turns


def _extract_usage(events: list[dict[str, Any]]) -> UsageStats:
    """从 transcript 事件中累加 token 用量。"""
    total_input = total_output = total_tokens = 0
    total_cost = 0.0

    for ev in events:
        if ev.get("type") != "message":
            continue
        msg = ev.get("message", {})
        usage = msg.get("usage", {})
        if usage:
            total_input += usage.get("input", usage.get("input_tokens", 0))
            total_output += usage.get("output", usage.get("output_tokens", 0))
            total_tokens += usage.get("totalTokens", usage.get("total_tokens", 0))
            cost = usage.get("cost", {})
            if isinstance(cost, dict):
                total_cost += cost.get("total", 0.0) or 0.0

    return UsageStats(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_tokens,
        cost_usd=total_cost,
    )


def transcript_to_sample(
    *,
    transcript_path: Path,
    task_id: str,
    prompt: str,
    grading_type: str,
    reward_terminal: float,
    reward_breakdown: dict[str, float],
    model_id: str,
    seed: int,
    run_index: int,
    execution_time: float = 0.0,
    timed_out: bool = False,
    workspace: str = "",
) -> TrainingSample:
    """将一个 openclaw transcript 转成 TrainingSample。"""
    events = _parse_openclaw_transcript(transcript_path)
    turns = _events_to_turns(events)
    usage = _extract_usage(events)
    split = split_for_seed(seed)
    sample_id = f"{task_id}-seed{seed}-run{run_index}"

    return TrainingSample(
        sample_id=sample_id,
        task_id=task_id,
        split=split,
        seed=seed,
        run_index=run_index,
        model_id=model_id,
        prompt=prompt,
        grading_type=grading_type,
        trajectory=turns,
        reward=Reward(terminal=reward_terminal, breakdown=reward_breakdown),
        usage=usage,
        execution_time=execution_time,
        timed_out=timed_out,
        workspace=workspace,
    )


def results_to_samples(
    results_json: Path,
    transcripts_dir: Path,
    model_id: str,
    seed: int = 0,
) -> list[TrainingSample]:
    """从一次完整 benchmark run 的结果里批量转换所有 task 的 TrainingSample。

    Args:
        results_json: PinchBench 输出的 results JSON 文件
        transcripts_dir: 对应的 transcripts 目录
        model_id: 采样模型 ID
        seed: 本次 run 的 seed（用于 split 划分）
    """
    data = json.loads(results_json.read_text(encoding="utf-8"))
    samples = []

    for task_entry in data.get("tasks", []):
        task_id = task_entry["task_id"]
        transcript_path = transcripts_dir / f"{task_id}.jsonl"
        if not transcript_path.exists():
            continue

        grading = task_entry.get("grading", {})
        runs = grading.get("runs", [])
        if not runs:
            continue

        # 取第一次 run 的 grading 结果（单次采样）
        run0 = runs[0]
        reward_terminal = float(run0.get("score", 0.0))
        reward_breakdown = {
            k: float(v)
            for k, v in run0.get("breakdown", {}).items()
        }

        fm = task_entry.get("frontmatter", {})
        prompt = fm.get("prompt", "")
        grading_type = fm.get("grading_type", "automated")

        sample = transcript_to_sample(
            transcript_path=transcript_path,
            task_id=task_id,
            prompt=prompt,
            grading_type=grading_type,
            reward_terminal=reward_terminal,
            reward_breakdown=reward_breakdown,
            model_id=model_id,
            seed=seed,
            run_index=0,
            execution_time=task_entry.get("execution_time", 0.0),
            timed_out=task_entry.get("timed_out", False),
            workspace=task_entry.get("workspace", ""),
        )
        samples.append(sample)

    return samples


def build_grpo_group(samples: list[TrainingSample]) -> GRPOGroup:
    """把同一 task+seed 的多次采样打包成 GRPOGroup。"""
    assert samples, "samples 不能为空"
    task_id = samples[0].task_id
    seed = samples[0].seed
    prompt = samples[0].prompt
    assert all(s.task_id == task_id and s.seed == seed for s in samples), \
        "GRPOGroup 要求所有 sample 来自同一 task+seed"
    return GRPOGroup(task_id=task_id, seed=seed, prompt=prompt, samples=samples)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法: python convert.py <results_json> <transcripts_dir> [output.jsonl]")
        sys.exit(1)

    results_json = Path(sys.argv[1])
    transcripts_dir = Path(sys.argv[2])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("rl_samples.jsonl")

    # 从 results JSON 文件名推断 model_id
    stem = results_json.stem  # e.g. "0004_qwen-plus"
    model_id = stem.split("_", 1)[1] if "_" in stem else stem

    samples = results_to_samples(
        results_json=results_json,
        transcripts_dir=transcripts_dir,
        model_id=model_id,
        seed=0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    print(f"转换完成：{len(samples)} 条 sample → {output_path}")
