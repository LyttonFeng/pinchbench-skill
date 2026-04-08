"""
PinchBench RL 采样脚本。

对指定 task（或 tasks-dir 下所有 task）执行多次 openclaw agent，
每次独立 workspace，收集 transcript，经 convert.py 转成 TrainingSample，写入 JSONL。

Live-User 场景：同一个 task 跑多次 = 用户反复发相似请求，模型边用边学。
--runs 控制每个 task 的采样次数，seed 自动递增保证 sample_id 不重复。

用法：
    # 配置 openclaw 接 vLLM（首次运行）
    python rl/collect.py --setup \
        --base-url http://<runpod-ip>:8000/v1 \
        --model Qwen/Qwen3-4B

    # 对 2 个训练 task 各采样 30 次（共 60 条 trajectory）
    python rl/collect.py \
        --tasks-dir tasks \
        --task-ids task_18_market_research task_21_openclaw_comprehension \
        --runs 30 \
        --base-url http://66.92.198.162:8000/v1 \
        --model Qwen/Qwen3-4B \
        --judge-model qwen-plus \
        --judge-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --judge-api-key sk-xxx \
        --output rl/data/samples_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# 把 scripts/ 加进 import 路径，复用 benchmark 的库
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from lib_agent import ensure_agent_exists, execute_openclaw_task, cleanup_agent_sessions
from lib_grading import grade_task
from lib_tasks import TaskLoader, Task
from convert import transcript_to_sample
from schema import split_for_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("collect")

_RL_AGENT_ID = "rl-collect-agent"
_TRANSCRIPT_TMP_DIR = Path("/tmp/pinchbench_rl/transcripts")


def _setup_openclaw(base_url: str, model: str, api_key: str | None) -> None:
    """配置 openclaw 接 vLLM endpoint。每次强制重建 agent 确保模型配置生效。"""
    import json
    import subprocess

    logger.info("配置 openclaw agent: %s -> %s", _RL_AGENT_ID, base_url)
    workspace = Path("/tmp/pinchbench_rl/workspace")
    workspace.mkdir(parents=True, exist_ok=True)

    # 动态更新 main agent 的 models.json，把 vLLM provider 写进去
    # 这样 openclaw 才能通过 "vllm/<model>" 路由到正确的 endpoint
    main_models_path = Path.home() / ".openclaw" / "agents" / "main" / "agent" / "models.json"
    if main_models_path.exists():
        try:
            data = json.loads(main_models_path.read_text("utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}

    provider_name = "vllm"
    model_bare = model.split("/", 1)[-1] if "/" in model else model
    data.setdefault("providers", {})[provider_name] = {
        "baseUrl": base_url,
        "apiKey": api_key or "dummy",
        "api": "openai-completions",
        "models": [{
            "id": model_bare,
            "name": model_bare,
            "reasoning": False,
            "input": ["text"],
            "contextWindow": 32768,
            "maxTokens": 8192,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "api": "openai-completions",
        }],
    }
    main_models_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
    logger.info("已更新 main models.json: %s -> %s", provider_name, base_url)

    # 先删除旧 agent（忽略失败）
    subprocess.run(
        ["openclaw", "agents", "delete", _RL_AGENT_ID, "--force"],
        capture_output=True, check=False,
    )

    # model 格式：vllm/<model_bare>，openclaw 据此路由到 vllm provider
    routed_model = f"{provider_name}/{model_bare}"
    ensure_agent_exists(
        _RL_AGENT_ID,
        routed_model,
        workspace,
        base_url=base_url,
        api_key=api_key,
    )
    logger.info("openclaw agent 配置完成: %s (model=%s)", _RL_AGENT_ID, routed_model)


def _load_rl_tasks(tasks_dir: Path) -> list[Task]:
    """加载 rl/tasks/ 下的变体 task。"""
    if not tasks_dir.exists():
        logger.error("tasks 目录不存在: %s", tasks_dir)
        sys.exit(1)
    loader = TaskLoader(tasks_dir)
    tasks = loader.load_all_tasks()
    logger.info("加载到 %d 个 RL 训练 task", len(tasks))
    return tasks


def _load_single_task(task_path: Path) -> Task:
    """加载单个 task 文件。"""
    loader = TaskLoader(task_path.parent)
    tasks = loader.load_all_tasks()
    matched = [t for t in tasks if t.task_id in task_path.stem]
    if not matched:
        # fallback: 加载第一个
        matched = tasks
    if not matched:
        logger.error("无法加载 task: %s", task_path)
        sys.exit(1)
    return matched[0]


def collect_one(
    task: Task,
    model_id: str,
    base_url: str,
    api_key: str | None,
    skill_dir: Path,
    seed: int,
    run_index: int,
    output_path: Path,
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> bool:
    """执行一个 task，采集 transcript，写入 TrainingSample。"""
    run_id = f"rl-{int(time.time())}"
    transcript_dir = _TRANSCRIPT_TMP_DIR / run_id
    transcript_dir.mkdir(parents=True, exist_ok=True)

    logger.info("采样 task=%s seed=%d run=%d", task.task_id, seed, run_index)

    cleanup_agent_sessions(_RL_AGENT_ID)

    try:
        result = execute_openclaw_task(
            task=task,
            agent_id=_RL_AGENT_ID,
            model_id=model_id,
            run_id=run_id,
            timeout_multiplier=1.0,
            skill_dir=skill_dir,
            output_dir=transcript_dir,
            verbose=False,
        )
    except Exception as exc:
        logger.warning("task 执行失败 %s: %s", task.task_id, exc)
        return False

    # grading
    try:
        grade_kwargs: dict = dict(task=task, execution_result=result, skill_dir=skill_dir)
        if judge_model:
            grade_kwargs["judge_model"] = judge_model
            grade_kwargs["judge_backend"] = "api"
            if judge_base_url:
                grade_kwargs["judge_base_url"] = judge_base_url
            if judge_api_key:
                grade_kwargs["judge_api_key"] = judge_api_key
        grade = grade_task(**grade_kwargs)
    except Exception as exc:
        logger.warning("grading 失败 %s: %s", task.task_id, exc)
        return False

    transcript_path = transcript_dir / f"{task.task_id}.jsonl"
    if not transcript_path.exists():
        logger.warning("transcript 文件不存在: %s", transcript_path)
        return False

    # 转成 TrainingSample
    try:
        sample = transcript_to_sample(
            transcript_path=transcript_path,
            task_id=task.task_id,
            prompt=task.prompt,
            grading_type=task.grading_type,
            reward_terminal=grade.score,
            reward_breakdown=grade.breakdown,
            model_id=model_id,
            seed=seed,
            run_index=run_index,
            execution_time=result.get("execution_time", 0.0),
            timed_out=result.get("timed_out", False),
            workspace=result.get("workspace", ""),
        )
    except Exception as exc:
        logger.warning("transcript 转换失败 %s: %s", task.task_id, exc)
        return False

    # 写入 JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    split = split_for_seed(seed)
    logger.info(
        "✅ %s  reward=%.3f  split=%s  turns=%d",
        task.task_id,
        grade.score,
        split,
        len(sample.assistant_turns),
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="PinchBench RL 采样脚本")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="仅配置 openclaw agent，不采样",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path("rl/tasks"),
        help="RL 训练 task 目录（默认 rl/tasks/）",
    )
    parser.add_argument(
        "--task",
        type=Path,
        default=None,
        help="指定单个 task 文件路径",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="vLLM endpoint URL，如 http://<runpod-ip>:8000/v1",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="模型 ID，如 Qwen/Qwen3-4B",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key（vLLM 本地服务可留空或填任意字符串）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/data/samples_raw.jsonl"),
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="只采样指定 task_id（空格分隔），不传则采样所有 task",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="每个 task 的采样次数（默认 1，Live-User 场景建议 30）",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="起始 seed，之后每次采样自动 +1（决定 train/val/test split）",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="LLM judge 模型 ID，hybrid/llm_judge task 需要（如 qwen-plus）",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="LLM judge endpoint URL（如 https://dashscope.aliyuncs.com/compatible-mode/v1）",
    )
    parser.add_argument(
        "--judge-api-key",
        default=None,
        help="LLM judge API key",
    )
    args = parser.parse_args()

    skill_dir = _REPO_ROOT
    api_key = args.api_key or "dummy"  # vLLM 本地服务不需要真实 key

    # 配置 openclaw
    _setup_openclaw(args.base_url, args.model, api_key)

    if args.setup:
        logger.info("--setup 完成，退出")
        return

    # 加载 task
    if args.task:
        tasks = [_load_single_task(args.task)]
    else:
        tasks = _load_rl_tasks(args.tasks_dir)

    # 按 task_id 过滤
    if args.task_ids:
        tasks = [t for t in tasks if t.task_id in args.task_ids]
        if not tasks:
            logger.error("--task-ids 指定的 task 未找到: %s", args.task_ids)
            sys.exit(1)

    if not tasks:
        logger.error("没有找到可用的 task")
        sys.exit(1)

    total = len(tasks) * args.runs
    logger.info(
        "开始采样：%d 个 task × %d 次 = %d 条 trajectory",
        len(tasks), args.runs, total,
    )

    # 多次采样循环：每个 task 跑 --runs 次，seed 自动递增
    success = 0
    global_run = 0
    for task in tasks:
        for run_index in range(args.runs):
            seed = args.seed_start + global_run
            ok = collect_one(
                task=task,
                model_id=args.model,
                base_url=args.base_url,
                api_key=api_key,
                skill_dir=skill_dir,
                seed=seed,
                run_index=run_index,
                output_path=args.output,
                judge_model=args.judge_model,
                judge_base_url=args.judge_base_url,
                judge_api_key=args.judge_api_key,
            )
            if ok:
                success += 1
            global_run += 1

    logger.info("采样完成：%d/%d 成功 → %s", success, total, args.output)


if __name__ == "__main__":
    main()
