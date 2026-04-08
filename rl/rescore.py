"""
vLLM re-score：对 TrainingSample 的 trajectory 补全 logprobs。

openclaw 采样时不暴露 token logprobs，需要用 vLLM 对每个 assistant turn
做一次 forward pass，拿到每个 token 的 log probability。

用法（RunPod 上执行）：
    python rl/rescore.py \
        --input rl/data/samples_raw.jsonl \
        --output rl/data/samples_rescored.jsonl \
        --model Qwen/Qwen3-4B \
        --base-url http://localhost:8000/v1

原理：
    vLLM 的 /v1/chat/completions 支持 echo=True + logprobs=N，
    可以对给定的 prompt+completion 做 forward，返回每个 token 的 logprob。
    我们把 prompt（历史对话）和 completion（assistant 回复）拼好，
    调用 API 拿回 logprobs，写入 TurnMessage.logprobs。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

sys.path.insert(0, str(Path(__file__).parent))
from schema import TrainingSample, TurnMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rescore")


def _build_messages_up_to(
    trajectory: list[TurnMessage], until_index: int
) -> list[dict[str, Any]]:
    """构建到第 until_index 轮（不含）的 messages 列表，作为 prompt。"""
    messages = []
    for turn in trajectory[:until_index]:
        if turn.role == "user":
            messages.append({"role": "user", "content": turn.content})
        elif turn.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": turn.content}
            if turn.tool_calls:
                msg["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
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


def _rescore_turn(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt_messages: list[dict[str, Any]],
    completion_text: str,
    timeout: float = 60.0,
) -> list[float] | None:
    """
    对一个 assistant turn 做 re-score，返回每个 token 的 logprob。

    使用 vLLM 的 /v1/chat/completions 接口：
      - echo=True：把 prompt+completion 一起做 forward
      - logprobs=1：返回每个位置 top-1 的 logprob

    返回 completion 部分每个 token 的 logprob list，
    若失败返回 None。
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"

    # 把 completion 当成最后一条 assistant 消息，让 vLLM echo
    messages = prompt_messages + [
        {"role": "assistant", "content": completion_text}
    ]

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 1,        # 不生成新 token，只要 echo 部分的 logprobs
        "echo": True,           # vLLM 扩展参数：返回 prompt+completion 的 logprobs
        "logprobs": 1,          # 每个位置返回 top-1 logprob
        "temperature": 0.0,
    }).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    req = request.Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        logger.warning("re-score HTTP 错误 %s: %s", exc.code, body)
        return None
    except Exception as exc:
        logger.warning("re-score 请求失败: %s", exc)
        return None

    # 从响应中提取 completion 部分的 logprobs
    # vLLM echo 模式下，choices[0].logprobs.token_logprobs 包含 prompt+completion 所有 token
    try:
        choice = data["choices"][0]
        lp = choice.get("logprobs", {})
        token_logprobs = lp.get("token_logprobs", [])

        # vLLM 会返回 prompt 部分（None）+ completion 部分的 logprobs
        # 过滤掉 None（prompt 部分），只取 completion 部分
        completion_logprobs = [x for x in token_logprobs if x is not None]

        if not completion_logprobs:
            logger.warning("re-score 返回空 logprobs")
            return None

        return completion_logprobs
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("解析 re-score 响应失败: %s", exc)
        return None


def rescore_sample(
    sample: TrainingSample,
    base_url: str,
    model: str,
    api_key: str,
) -> TrainingSample:
    """对一条 TrainingSample 的所有 assistant turn 补全 logprobs。"""
    new_trajectory = []
    for i, turn in enumerate(sample.trajectory):
        if turn.role != "assistant":
            new_trajectory.append(turn)
            continue

        if turn.logprobs is not None:
            # 已有 logprobs，跳过
            new_trajectory.append(turn)
            continue

        # 构建 prompt（当前 turn 之前的所有消息）
        prompt_messages = _build_messages_up_to(sample.trajectory, i)
        completion = turn.content

        # 如果有 tool_calls，把它们序列化进 completion
        if turn.tool_calls:
            tool_calls_str = json.dumps(
                [{"name": tc.name, "arguments": tc.arguments} for tc in turn.tool_calls],
                ensure_ascii=False,
            )
            completion = f"{completion}\n[TOOL_CALLS]{tool_calls_str}".strip()

        logprobs = _rescore_turn(
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt_messages=prompt_messages,
            completion_text=completion,
        )

        new_turn = TurnMessage(
            role=turn.role,
            content=turn.content,
            tool_calls=turn.tool_calls,
            tool_name=turn.tool_name,
            logprobs=logprobs,
        )
        new_trajectory.append(new_turn)

        if logprobs:
            logger.debug(
                "  turn %d: %d tokens, logprob mean=%.3f",
                i,
                len(logprobs),
                sum(logprobs) / len(logprobs),
            )

    from dataclasses import replace
    return TrainingSample(
        sample_id=sample.sample_id,
        task_id=sample.task_id,
        split=sample.split,
        seed=sample.seed,
        run_index=sample.run_index,
        model_id=sample.model_id,
        prompt=sample.prompt,
        grading_type=sample.grading_type,
        trajectory=new_trajectory,
        reward=sample.reward,
        usage=sample.usage,
        execution_time=sample.execution_time,
        timed_out=sample.timed_out,
        workspace=sample.workspace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM re-score：补全 TrainingSample 的 logprobs")
    parser.add_argument("--input", type=Path, required=True, help="输入 JSONL（samples_raw）")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSONL（samples_rescored）")
    parser.add_argument("--model", required=True, help="模型 ID，如 Qwen/Qwen3-4B")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM endpoint URL（默认 http://localhost:8000/v1）",
    )
    parser.add_argument("--api-key", default="dummy", help="API key（vLLM 本地服务填 dummy）")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已有 logprobs 的 sample")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("输入文件不存在: %s", args.input)
        sys.exit(1)

    samples = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(TrainingSample.from_dict(json.loads(line)))

    logger.info("加载 %d 条 sample，开始 re-score", len(samples))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    success = 0

    with args.output.open("w", encoding="utf-8") as out_f:
        for i, sample in enumerate(samples, 1):
            if args.skip_existing and sample.has_logprobs:
                logger.info("[%d/%d] 跳过（已有 logprobs）: %s", i, len(samples), sample.sample_id)
                out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                success += 1
                continue

            logger.info("[%d/%d] re-score: %s", i, len(samples), sample.sample_id)
            rescored = rescore_sample(
                sample=sample,
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
            )
            out_f.write(json.dumps(rescored.to_dict(), ensure_ascii=False) + "\n")

            if rescored.has_logprobs:
                success += 1
                logger.info(
                    "  ✅ reward=%.3f  assistant_turns=%d",
                    rescored.reward.terminal,
                    len(rescored.assistant_turns),
                )
            else:
                logger.warning("  ⚠️  re-score 未能补全 logprobs: %s", sample.sample_id)

            time.sleep(0.1)  # 避免打爆 vLLM

    logger.info("re-score 完成：%d/%d 成功 → %s", success, len(samples), args.output)


if __name__ == "__main__":
    main()
