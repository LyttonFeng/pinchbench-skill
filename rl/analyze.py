"""
分析 PinchBench benchmark 结果，输出 RL 难度分布表。

用法：
    python rl/analyze.py results/0004_qwen-plus.json
    python rl/analyze.py results/0004_qwen-plus.json --jsonl rl/data/analysis.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# RL 价值分桶阈值
_HARD_MIN = 0.45    # all-fail 时 score >= 0.45 → hard（near-miss）
_MODERATE_MIN = 0.1  # all-fail 时 score >= 0.1 → moderate

_BUCKET_ORDER = ["hard", "moderate", "very_hard", "easy"]
_BUCKET_MEANING = {
    "hard":      "最佳 RL 目标——有梯度，有提升空间",
    "moderate":  "弱信号——部分得分但离通过还远",
    "very_hard": "当前模型太难——近零梯度，无有效信号",
    "easy":      "已掌握——RL 优先级低",
}


def _rl_bucket(score: float, grading_type: str) -> str:
    """根据得分和评分类型分配 RL 价值桶。"""
    if score >= 1.0:
        return "easy"
    if score == 0.0:
        return "very_hard"
    if score >= _HARD_MIN:
        return "hard"
    if score >= _MODERATE_MIN:
        return "moderate"
    return "very_hard"


def analyze(results_json: Path, output_jsonl: Path | None = None) -> None:
    data = json.loads(results_json.read_text(encoding="utf-8"))
    model = data.get("model", "unknown")
    run_id = data.get("run_id", "?")
    tasks = data.get("tasks", [])

    if not tasks:
        print("没有找到 task 数据")
        sys.exit(1)

    rows = []
    for task in tasks:
        task_id = task["task_id"]
        grading = task.get("grading", {})
        score = float(grading.get("mean", 0.0))
        fm = task.get("frontmatter", {})
        grading_type = fm.get("grading_type", "unknown")
        category = fm.get("category", "unknown")
        breakdown = {}
        runs = grading.get("runs", [])
        if runs:
            breakdown = runs[0].get("breakdown", {})

        bucket = _rl_bucket(score, grading_type)
        rows.append({
            "task_id": task_id,
            "category": category,
            "grading_type": grading_type,
            "score": round(score, 4),
            "rl_bucket": bucket,
            "breakdown": breakdown,
            "execution_time": task.get("execution_time", 0.0),
            "timed_out": task.get("timed_out", False),
        })

    # 按 bucket 优先级 + score 排序
    priority = {b: i for i, b in enumerate(_BUCKET_ORDER)}
    rows.sort(key=lambda r: (priority.get(r["rl_bucket"], 9), r["score"]))

    # 打印表格
    print(f"\nPinchBench RL 分析 — model={model}  run={run_id}")
    print(f"{'='*90}")
    print(f"{'TASK_ID':<45} {'SCORE':>6} {'TYPE':<12} {'BUCKET':<12} CATEGORY")
    print(f"{'-'*90}")
    for r in rows:
        marker = "★ " if r["rl_bucket"] == "hard" else "  "
        print(
            f"{marker}{r['task_id']:<43} "
            f"{r['score']:>6.3f} "
            f"{r['grading_type']:<12} "
            f"{r['rl_bucket']:<12} "
            f"{r['category']}"
        )

    # 分桶汇总
    from collections import defaultdict
    by_bucket: dict[str, list] = defaultdict(list)
    for r in rows:
        by_bucket[r["rl_bucket"]].append(r)

    n = len(rows)
    mean_score = sum(r["score"] for r in rows) / n if n else 0.0

    print(f"\n{'='*90}")
    print("分桶汇总")
    print(f"{'-'*90}")
    print(f"{'BUCKET':<12} {'任务数':>6} {'占比':>7} {'平均分':>8}  说明")
    print(f"{'-'*90}")
    for bucket in _BUCKET_ORDER:
        bucket_rows = by_bucket.get(bucket, [])
        k = len(bucket_rows)
        pct = 100.0 * k / n if n else 0.0
        avg = sum(r["score"] for r in bucket_rows) / k if k else 0.0
        meaning = _BUCKET_MEANING.get(bucket, "")
        print(f"{bucket:<12} {k:>6} {pct:>6.1f}% {avg:>8.3f}  {meaning}")
    print(f"{'-'*90}")
    print(f"{'合计':<12} {n:>6} {'100.0%':>7} {mean_score:>8.3f}")

    # 重点输出 hard bucket
    hard = by_bucket.get("hard", [])
    if hard:
        print(f"\n{'='*90}")
        print(f"★ Hard bucket — 优先 RL 训练目标（{len(hard)} 个任务）")
        print(f"{'-'*90}")
        for r in hard:
            bd = "  |  ".join(f"{k}={v:.2f}" for k, v in r["breakdown"].items())
            print(f"  {r['task_id']:<45} score={r['score']:.3f}  {r['grading_type']}")
            if bd:
                print(f"    breakdown: {bd}")

    # 写 JSONL
    if output_jsonl:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n完整分析结果写入: {output_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 PinchBench 结果，输出 RL 难度分布")
    parser.add_argument("results_json", help="PinchBench results JSON 文件路径")
    parser.add_argument("--jsonl", default=None, help="输出完整分析结果到 JSONL 文件")
    args = parser.parse_args()

    results_json = Path(args.results_json)
    if not results_json.exists():
        print(f"文件不存在: {results_json}")
        sys.exit(1)

    output_jsonl = Path(args.jsonl) if args.jsonl else None
    analyze(results_json, output_jsonl)


if __name__ == "__main__":
    main()
