#!/usr/bin/env python3
"""Compare two PinchBench benchmark.py JSON outputs; highlight RL training 8 tasks.

Usage:
  python3 rl/scripts/compare_benchmark_json.py \\
    results/bench/foo_baseline.json results/bench/foo_lora.json \\
    --out-dir results/bench/compare_20260411
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Canonical task_id from tasks/*.md frontmatter (matches prepare_prompts DEFAULT_TASK_IDS)
RL_TRAIN_8 = frozenset(
    {
        "task_02_stock",
        "task_10_workflow",
        "task_12_skill_search",
        "task_16_email_triage",
        "task_18_market_research",
        "task_18_spreadsheet_summary",
        "task_22_second_brain",
        "task_24_polymarket_briefing",
    }
)


def _mean_score(task_entry: dict) -> float | None:
    g = task_entry.get("grading") or {}
    if isinstance(g.get("mean"), (int, float)):
        return float(g["mean"])
    runs = g.get("runs") or []
    if runs and isinstance(runs[0], dict) and "score" in runs[0]:
        return float(runs[0]["score"])
    return None


def _load_scores(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id: dict[str, float] = {}
    for row in data.get("tasks") or []:
        tid = row.get("task_id")
        if not tid:
            continue
        m = _mean_score(row)
        if m is None:
            continue
        by_id[tid] = m
    return by_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_json", type=Path)
    ap.add_argument("lora_json", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    base = _load_scores(args.baseline_json)
    lora = _load_scores(args.lora_json)
    all_ids = sorted(set(base) | set(lora))

    rows = []
    for tid in all_ids:
        b = base.get(tid)
        l = lora.get(tid)
        delta = None if b is None or l is None else l - b
        rows.append(
            {
                "task_id": tid,
                "rl_train_8": tid in RL_TRAIN_8,
                "score_baseline": b,
                "score_lora": l,
                "delta": delta,
            }
        )

    r8 = [r for r in rows if r["rl_train_8"]]
    oth = [r for r in rows if not r["rl_train_8"]]

    def _avg(xs: list[dict]) -> tuple[float, int]:
        ds = [r["delta"] for r in xs if r["delta"] is not None]
        if not ds:
            return 0.0, 0
        return sum(ds) / len(ds), len(ds)

    avg8, n8 = _avg(r8)
    avgo, no = _avg(oth)

    summary = {
        "baseline_file": str(args.baseline_json),
        "lora_file": str(args.lora_json),
        "rl_train_8_task_ids": sorted(RL_TRAIN_8),
        "mean_delta_rl_train_8": avg8,
        "tasks_with_delta_rl_train_8": n8,
        "mean_delta_other": avgo,
        "tasks_with_delta_other": no,
        "per_task": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jpath = args.out_dir / "compare.json"
    mpath = args.out_dir / "compare.md"

    jpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# PinchBench baseline vs LoRA",
        "",
        f"- Baseline: `{args.baseline_json}`",
        f"- LoRA: `{args.lora_json}`",
        "",
        "## Summary",
        "",
        f"- **RL 训练 8 任务**（均值 Δ = LoRA − baseline）: **{avg8:+.4f}** （{n8} 个有对比）",
        f"- **其余任务** 均值 Δ: **{avgo:+.4f}** （{no} 个有对比）",
        "",
        "### RL 训练 8（关注是否明显提升）",
        "",
        "| task_id | baseline | LoRA | Δ |",
        "|---------|----------|------|---|",
    ]
    for r in sorted(r8, key=lambda x: x["task_id"]):
        b = r["score_baseline"]
        l = r["score_lora"]
        d = r["delta"]
        lines.append(
            f"| {r['task_id']} | {b if b is not None else '—'} | {l if l is not None else '—'} | {d if d is not None else '—'} |"
        )
    lines.extend(["", "### 其余任务（希望不明显下降）", "", "| task_id | baseline | LoRA | Δ |", "|---------|----------|------|---|"])
    for r in sorted(oth, key=lambda x: x["task_id"]):
        b = r["score_baseline"]
        l = r["score_lora"]
        d = r["delta"]
        lines.append(
            f"| {r['task_id']} | {b if b is not None else '—'} | {l if l is not None else '—'} | {d if d is not None else '—'} |"
        )
    lines.append("")
    mpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {jpath}")
    print(f"Wrote {mpath}")


if __name__ == "__main__":
    main()
