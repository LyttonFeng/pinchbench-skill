"""
PinchBench RL 训练数据格式定义。

一条 TrainingSample 对应一次完整的 task 执行（一个 episode）。
多条 sample 组成一个 GRPO group（同一 prompt，K 次采样）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnMessage:
    """对话中的一轮消息。

    role:
      - "user"      : 用户/系统发给 agent 的消息
      - "assistant" : agent 的回复（可能包含 tool_calls）
      - "tool"      : tool 执行结果

    logprobs: assistant turn 的每个 token 的 log prob。
      - GRPO 训练必需，采样时由 vLLM 记录
      - 若用 openclaw 采样（无法拿到 logprobs），设为 None
      - 后续可用 vLLM 离线 re-score 补全
    """
    role: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: Optional[str] = None       # role == "tool" 时填写
    logprobs: Optional[list[float]] = None  # role == "assistant" 时填写

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ]
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.logprobs is not None:
            d["logprobs"] = self.logprobs
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TurnMessage":
        tool_calls = [
            ToolCall(name=tc["name"], arguments=tc.get("arguments", {}))
            for tc in d.get("tool_calls", [])
        ]
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            tool_calls=tool_calls,
            tool_name=d.get("tool_name"),
            logprobs=d.get("logprobs"),
        )


@dataclass
class Reward:
    """Terminal reward，来自 PinchBench grading 函数。

    terminal: grading 函数输出的 [0, 1] 分数，作为 GRPO 的 reward。
    breakdown: 各评分维度的细项分数，用于分析和 debug。
    """
    terminal: float
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"terminal": self.terminal, "breakdown": self.breakdown}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Reward":
        return cls(terminal=d["terminal"], breakdown=d.get("breakdown", {}))


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UsageStats":
        return cls(
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
            cost_usd=d.get("cost_usd", 0.0),
        )


# seed 范围 → split 映射（与 clawgym 保持一致）
_SPLIT_BANDS = [
    (0, 10_000, "train"),
    (10_000, 11_000, "val"),
    (11_000, 12_000, "test"),
]


def split_for_seed(seed: int) -> str:
    """根据 seed 返回 train/val/test/ood。"""
    for lo, hi, name in _SPLIT_BANDS:
        if lo <= seed < hi:
            return name
    return "ood"


@dataclass
class TrainingSample:
    """一次完整的 task 执行，对应 GRPO group 中的一条 trajectory。

    sample_id 格式：{task_id}-seed{seed}-run{run_index}
    """
    sample_id: str
    task_id: str
    split: str                      # train / val / test / ood
    seed: int                       # task 变体的生成 seed
    run_index: int                  # 同一 task 第几次采样（GRPO group 内的 index）
    model_id: str
    prompt: str                     # 发给 agent 的完整 prompt（task.prompt）
    grading_type: str               # automated / llm_judge / hybrid
    trajectory: list[TurnMessage]
    reward: Reward
    usage: UsageStats = field(default_factory=UsageStats)
    execution_time: float = 0.0
    timed_out: bool = False
    workspace: str = ""             # 执行时的 workspace 路径（调试用）

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "split": self.split,
            "seed": self.seed,
            "run_index": self.run_index,
            "model_id": self.model_id,
            "prompt": self.prompt,
            "grading_type": self.grading_type,
            "trajectory": [t.to_dict() for t in self.trajectory],
            "reward": self.reward.to_dict(),
            "usage": self.usage.to_dict(),
            "execution_time": self.execution_time,
            "timed_out": self.timed_out,
            "workspace": self.workspace,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingSample":
        return cls(
            sample_id=d["sample_id"],
            task_id=d["task_id"],
            split=d["split"],
            seed=d["seed"],
            run_index=d["run_index"],
            model_id=d["model_id"],
            prompt=d["prompt"],
            grading_type=d["grading_type"],
            trajectory=[TurnMessage.from_dict(t) for t in d["trajectory"]],
            reward=Reward.from_dict(d["reward"]),
            usage=UsageStats.from_dict(d.get("usage", {})),
            execution_time=d.get("execution_time", 0.0),
            timed_out=d.get("timed_out", False),
            workspace=d.get("workspace", ""),
        )


@dataclass
class GRPOGroup:
    """同一 task prompt 的 K 次采样，构成一个 GRPO 训练组。

    samples 里的所有 sample 来自同一 task（相同 task_id + seed），
    prompt 相同，reward 不同，用于组内对比计算 advantage。
    """
    task_id: str
    seed: int
    prompt: str
    samples: list[TrainingSample]

    @property
    def rewards(self) -> list[float]:
        return [s.reward.terminal for s in self.samples]

    @property
    def split(self) -> str:
        return self.samples[0].split if self.samples else "ood"

    def advantages(self) -> list[float]:
        """GRPO 组内归一化 advantage。"""
        import statistics
        r = self.rewards
        if len(r) <= 1:
            return [0.0] * len(r)
        mu = statistics.mean(r)
        sigma = statistics.stdev(r)
        if sigma < 1e-8:
            return [0.0] * len(r)
        return [(x - mu) / sigma for x in r]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "prompt": self.prompt,
            "rewards": self.rewards,
            "advantages": self.advantages(),
            "samples": [s.to_dict() for s in self.samples],
        }
