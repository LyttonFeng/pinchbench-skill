#!/usr/bin/env python3
"""Collect runtime rollouts for synthetic spreadsheet DPO data.

This is the train/infer-parity path for task_18_spreadsheet_summary:

1. Load synthetic variants produced by build_spreadsheet_dataset.py.
2. Materialize each variant's real workspace files.
3. Run the model through the actual OpenClaw runtime.
4. Dynamically grade against that variant's expected answer.
5. Save transcript-backed rollout JSONL for build_dpo_pairs.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_agent import ensure_agent_exists, execute_openclaw_task, slugify_model  # noqa: E402
from lib_tasks import Task  # noqa: E402


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _safe_copy_workspace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _prepare_skill_assets(src_workspace: Path, skill_dir: Path) -> list[dict[str, str]]:
    assets = skill_dir / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)
    workspace_files: list[dict[str, str]] = []
    for item in sorted(src_workspace.iterdir()):
        if not item.is_file():
            continue
        shutil.copy2(item, assets / item.name)
        workspace_files.append({"source": item.name, "dest": item.name})
    return workspace_files


def _task_from_row(row: dict[str, Any], workspace_files: list[dict[str, str]], timeout_seconds: int) -> Task:
    return Task(
        task_id=row["task_id"],
        name="Synthetic Spreadsheet Summary",
        category="data_analysis",
        grading_type="automated",
        timeout_seconds=timeout_seconds,
        workspace_files=workspace_files,
        prompt=row["prompt"],
        expected_behavior="Read CSV and XLSX, compute statistics, write data_summary.md.",
        grading_criteria=[],
        automated_checks=None,
        llm_judge_rubric=None,
        grading_weights=None,
        file_path=None,
        frontmatter={},
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "thinking":
                parts.append(str(item.get("thinking", "")))
    return "\n".join(parts)


def _strip_workspace_path(text: str, workspace: str) -> str:
    """Replace absolute workspace/run paths with '.' so paths generalise at inference."""
    if not workspace:
        return text
    # Strip agent_workspace and its parent (run_dir) so both forms are cleaned:
    #   /tmp/.../9100/agent_workspace/foo.csv  → foo.csv
    #   /tmp/.../9100/                          → ./
    for path in [workspace, str(Path(workspace).parent)]:
        p = path.rstrip("/")
        text = text.replace(p + "/", "")   # prefix with trailing slash → relative
        text = text.replace(p, ".")        # bare path → '.'
    return text


def _transcript_to_messages(
    transcript: list[dict[str, Any]],
    workspace: str = "",
) -> list[dict[str, Any]]:
    """Convert OpenClaw events to a compact OpenAI-style message list."""
    messages: list[dict[str, Any]] = []
    for event in transcript:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        role = msg.get("role")
        content = msg.get("content", [])
        if role == "user":
            text = _content_text(content).strip()
            if text:
                messages.append({"role": "user", "content": text})
        elif role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for item in content if isinstance(content, list) else []:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif item.get("type") == "thinking":
                    text_parts.append(f"<think>{item.get('thinking', '')}</think>")
                elif item.get("type") == "toolCall":
                    raw_args = json.dumps(item.get("arguments", {}), ensure_ascii=False)
                    clean_args = _strip_workspace_path(raw_args, workspace)
                    tool_calls.append({
                        "id": item.get("id") or item.get("toolCallId") or f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": clean_args,
                        },
                    })
            out: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts).strip()}
            if tool_calls:
                out["tool_calls"] = tool_calls
            messages.append(out)
        elif role == "toolResult":
            text = _strip_workspace_path(_content_text(content), workspace)
            messages.append({
                "role": "tool",
                "tool_call_id": msg.get("toolCallId", ""),
                "name": msg.get("toolName", ""),
                "content": text[:8000],
            })
    return messages


def _numbers_from_text(text: str) -> list[float]:
    values: list[float] = []
    for raw in re.findall(r"(?<![A-Za-z])[-+]?\$?\d[\d,]*(?:\.\d+)?", text):
        try:
            values.append(float(raw.replace("$", "").replace(",", "")))
        except ValueError:
            pass
    return values


def _has_number(text: str, expected: float | int, pct_tol: float = 1.0, abs_tol: float = 1.0) -> bool:
    target = float(expected)
    tol = max(abs_tol, abs(target) * pct_tol / 100.0)
    return any(abs(v - target) <= tol for v in _numbers_from_text(text))


def _has_text(text: str, value: str) -> bool:
    return value.lower() in text.lower()


def _tool_call_items(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for event in transcript:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") != "assistant":
            continue
        for item in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
            if isinstance(item, dict) and item.get("type") == "toolCall":
                calls.append(item)
    return calls


def _count_xlsx_read_calls(transcript: list[dict[str, Any]]) -> int:
    count = 0
    for item in _tool_call_items(transcript):
        if item.get("name") != "read":
            continue
        args = item.get("arguments", {})
        raw = json.dumps(args, ensure_ascii=False).lower()
        if "company_expenses.xlsx" in raw or ".xlsx" in raw:
            count += 1
    return count


def _used_execute_after_xlsx(transcript: list[dict[str, Any]]) -> bool:
    saw_xlsx_read = False
    for item in _tool_call_items(transcript):
        name = str(item.get("name", "")).lower()
        raw = json.dumps(item.get("arguments", {}), ensure_ascii=False).lower()
        if name == "read" and ".xlsx" in raw:
            saw_xlsx_read = True
        if saw_xlsx_read and name in {"execute", "exec", "bash", "shell", "run"}:
            return True
    return False


def _grade_runtime_rollout(row: dict[str, Any], workspace: Path, transcript: list[dict[str, Any]]) -> dict[str, Any]:
    expected = row["expected"]
    csv = expected["csv"]
    excel = expected["excel"]
    report_path = workspace / expected.get("target_file", "data_summary.md")
    if not report_path.exists():
        for alt in ["summary.md", "report.md", "data_report.md", "analysis.md"]:
            candidate = workspace / alt
            if candidate.exists():
                report_path = candidate
                break

    report = ""
    if report_path.exists():
        try:
            report = report_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            report = ""

    xlsx_reads = _count_xlsx_read_calls(transcript)
    breakdown = {
        "target_file_exists": 1.0 if report_path.exists() else 0.0,
        "csv_total_revenue_correct": 1.0 if _has_number(report, csv["total_revenue"]) else 0.0,
        "csv_total_profit_correct": 1.0 if _has_number(report, csv["total_profit"]) else 0.0,
        "csv_total_units_correct": 1.0 if _has_number(report, csv["total_units_sold"], abs_tol=0.0) else 0.0,
        "csv_top_region_correct": 1.0 if _has_text(report, csv["top_region_by_revenue"]) else 0.0,
        "csv_top_product_correct": 1.0 if _has_text(report, csv["top_product_by_revenue"]) else 0.0,
        "excel_total_q1_expenses_correct": 1.0 if _has_number(report, excel["total_q1_expenses"]) else 0.0,
        "excel_top_department_correct": 1.0 if _has_text(report, excel["department_highest_expenses"]) else 0.0,
        "excel_top_employee_correct": 1.0 if _has_text(report, excel["employee_highest_expenses"]) else 0.0,
        "budget_comparison_present": 1.0 if (
            "budget" in report.lower()
            and "actual" in report.lower()
            and any(_has_text(report, dept) for dept in excel["budget_comparison"])
        ) else 0.0,
        "no_repeated_xlsx_binary_read": 1.0 if xlsx_reads <= 1 else 0.0,
        "used_execute_after_xlsx": 1.0 if _used_execute_after_xlsx(transcript) else 0.0,
    }
    core_keys = [
        "target_file_exists",
        "csv_total_revenue_correct",
        "csv_total_profit_correct",
        "csv_total_units_correct",
        "csv_top_region_correct",
        "csv_top_product_correct",
        "excel_total_q1_expenses_correct",
        "excel_top_department_correct",
        "excel_top_employee_correct",
        "budget_comparison_present",
    ]
    score = sum(breakdown[k] for k in core_keys) / len(core_keys)
    if xlsx_reads > 1:
        score = min(score, 0.5)
    if not report_path.exists():
        score = min(score, 0.1)
    return {
        "score": round(float(score), 4),
        "breakdown": breakdown,
        "report_path": str(report_path) if report_path.exists() else None,
        "xlsx_read_calls": xlsx_reads,
    }


def collect(args: argparse.Namespace) -> None:
    manifest_path = Path(args.input)
    data_root = manifest_path.parent
    out_path = Path(args.output)
    transcript_dir = Path(args.transcript_dir)
    run_root = Path(args.run_root)
    rows = _load_jsonl(manifest_path, limit=args.limit)

    rollouts: list[dict[str, Any]] = []
    model_slug = slugify_model(args.model)
    agent_id = args.agent_id or f"spreadsheet-dpo-{args.role}-{model_slug}"
    os.environ["PINCHBENCH_RUN_ROOT"] = str(run_root)

    for idx, row in enumerate(rows):
        variant_id = row["variant_id"]
        src_workspace = data_root / row["workspace_dir"]
        run_id = f"{args.run_id_start + idx}-{args.role}-{model_slug}"
        workspace = run_root / str(args.run_id_start + idx) / "agent_workspace"
        skill_dir = run_root / "runtime_skill_assets" / args.role / model_slug / variant_id
        workspace_files = _prepare_skill_assets(src_workspace, skill_dir)
        task = _task_from_row(row, workspace_files, int(args.timeout_seconds))

        ensure_agent_exists(
            agent_id,
            args.model,
            workspace,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        result = execute_openclaw_task(
            task=task,
            agent_id=agent_id,
            model_id=args.model,
            run_id=run_id,
            timeout_multiplier=1.0,
            skill_dir=skill_dir,
            output_dir=None,
            verbose=args.verbose,
        )
        transcript = result.get("transcript", [])
        grade = _grade_runtime_rollout(row, workspace, transcript)

        transcript_path = transcript_dir / args.role / model_slug / f"{variant_id}.jsonl"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with transcript_path.open("w", encoding="utf-8") as f:
            for event in transcript:
                f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

        rollout = {
            "variant_id": variant_id,
            "task_id": row["task_id"],
            "split": row["split"],
            "model": args.model,
            "role": args.role,
            "score": grade["score"],
            "assistant_turns": sum(
                1
                for event in transcript
                if event.get("type") == "message"
                and event.get("message", {}).get("role") == "assistant"
            ),
            "status": result.get("status"),
            "workspace": str(workspace),
            "transcript_path": str(transcript_path),
            "messages": _transcript_to_messages(transcript, workspace=str(workspace)),
            "grade": grade,
            "execution_time": result.get("execution_time"),
            "exit_code": result.get("exit_code"),
            "timed_out": result.get("timed_out"),
            "stderr": (result.get("stderr") or "")[:2000],
        }
        rollouts.append(rollout)
        _write_jsonl(out_path, rollouts)
        print(json.dumps({
            "idx": idx,
            "variant_id": variant_id,
            "model": args.model,
            "role": args.role,
            "status": rollout["status"],
            "score": rollout["score"],
            "assistant_turns": rollout["assistant_turns"],
            "xlsx_read_calls": grade["xlsx_read_calls"],
        }, ensure_ascii=False))

    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "input": str(manifest_path),
        "output": str(out_path),
        "transcript_dir": str(transcript_dir),
        "role": args.role,
        "model": args.model,
        "base_url": args.base_url,
        "count": len(rollouts),
        "mean_score": round(sum(float(r["score"]) for r in rollouts) / len(rollouts), 4) if rollouts else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="rl_train.jsonl / rl_val.jsonl")
    parser.add_argument("--output", required=True, help="rollout JSONL path")
    parser.add_argument("--role", required=True, choices=["teacher", "student"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--run-root", default="/tmp/pinchbench_spreadsheet_dpo_runtime")
    parser.add_argument("--run-id-start", type=int, default=9100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--transcript-dir",
        default="rl/data/generated/task_18_spreadsheet_summary/runtime_transcripts",
    )
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
