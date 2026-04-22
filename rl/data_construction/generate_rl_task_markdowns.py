#!/usr/bin/env python3
"""Generate per-variant task markdowns + assets for RL training.

For each synthetic variant in rl_train.jsonl:
  1. Copies CSV/XLSX into assets/ with variant-specific names
  2. Generates a task markdown in tasks/ with:
     - variant-specific workspace_files
     - variant-specific automated_checks (regex on expected numbers)
     - the variant prompt

Usage:
    python3 rl/data_construction/generate_rl_task_markdowns.py \
        --input rl/data/generated/task_18_spreadsheet_summary_runtime/rl_train.jsonl \
        --workspace-base rl/data/generated/task_18_spreadsheet_summary_runtime \
        --tasks-dir tasks \
        --assets-dir assets \
        --prefix rl_ss
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def _num_pattern(value: float | int, pct_tol: float = 1.0) -> str:
    """Regex that matches a number within pct_tol% of value."""
    v = float(value)
    lo = v * (1 - pct_tol / 100)
    hi = v * (1 + pct_tol / 100)

    def fmt(n: float) -> str:
        if n == int(n):
            return str(int(n))
        return f"{n:.2f}"

    lo_s, hi_s = fmt(lo), fmt(hi)
    if lo_s == hi_s:
        digits = re.sub(r"[,.]", r"[,.]?", lo_s.replace(",", ""))
        return digits
    # Simple: just anchor on the rounded int
    rounded = int(round(v))
    s = f"{rounded:,}"
    return re.sub(r",", r"[,.]?", re.escape(s).replace(r"\,", ","))


def _make_automated_checks(csv: dict, excel: dict) -> str:
    total_revenue = int(round(float(csv["total_revenue"])))
    total_profit = int(round(float(csv["total_profit"])))
    total_units = int(csv["total_units_sold"])
    top_region = csv["top_region_by_revenue"]
    top_product = csv["top_product_by_revenue"]
    total_expenses = int(round(float(excel["total_q1_expenses"])))
    top_dept = excel["department_highest_expenses"]
    top_employee = excel["employee_highest_expenses"]
    budget_depts = excel.get("budget_comparison", [])

    rev_pat = _num_pattern(total_revenue)
    profit_pat = _num_pattern(total_profit)
    expense_pat = _num_pattern(total_expenses)
    region_lc = top_region.lower()
    product_lc = top_product.lower()
    dept_lc = top_dept.lower()
    # employee: match first + last name flexibly
    emp_parts = top_employee.lower().split()
    emp_pat = r"\s*".join(re.escape(p) for p in emp_parts)

    checks = f'''def grade(transcript: list, workspace_path: str) -> dict:
    from pathlib import Path
    import re

    scores = {{}}
    workspace = Path(workspace_path)
    report_path = workspace / "data_summary.md"
    if not report_path.exists():
        for alt in ["summary.md", "report.md", "data_report.md", "analysis.md"]:
            alt_path = workspace / alt
            if alt_path.exists():
                report_path = alt_path
                break

    if not report_path.exists():
        for k in ["report_created","total_revenue","total_profit","top_region",
                  "top_product","total_expenses","top_department","top_employee","budget_comparison"]:
            scores[k] = 0.0
        return scores

    scores["report_created"] = 1.0
    content = report_path.read_text(errors="replace")
    content_lower = content.lower()
    cn = content.replace(" ", "").replace(",", "").replace(".", "")

    # CSV: total revenue ~{total_revenue}
    scores["total_revenue"] = 1.0 if re.search(r"{rev_pat}", content.replace(" ","")) else 0.0

    # CSV: total profit ~{total_profit}
    scores["total_profit"] = 1.0 if re.search(r"{profit_pat}", content.replace(" ","")) else 0.0

    # CSV: top region = {top_region}
    scores["top_region"] = 1.0 if re.search(
        r"{region_lc}.*(?:top|highest|most|best|leading|largest)|(?:top|highest|most|best|leading|largest).*{region_lc}",
        content_lower) else 0.0

    # CSV: top product = {top_product}
    scores["top_product"] = 1.0 if re.search(
        r"{product_lc}.*(?:top|highest|most|best|leading|largest)|(?:top|highest|most|best|leading|largest).*{product_lc}",
        content_lower) else 0.0

    # Excel: total Q1 expenses ~{total_expenses}
    scores["total_expenses"] = 1.0 if re.search(r"{expense_pat}", content.replace(" ","")) else 0.0

    # Excel: top department = {top_dept}
    scores["top_department"] = 1.0 if re.search(
        r"{dept_lc}.*(?:top|highest|most|largest|leading)|(?:top|highest|most|largest|leading).*{dept_lc}",
        content_lower) else 0.0

    # Excel: top employee = {top_employee}
    scores["top_employee"] = 1.0 if re.search(r"{emp_pat}", content_lower) else 0.0

    # Excel: budget comparison present
    scores["budget_comparison"] = 1.0 if (
        "budget" in content_lower and "actual" in content_lower
    ) else 0.0

    return scores
'''
    return checks


def _make_markdown(variant_id: str, prompt: str, csv: dict, excel: dict,
                   csv_asset: str, xlsx_asset: str) -> str:
    total_revenue = int(round(float(csv["total_revenue"])))
    total_profit = int(round(float(csv["total_profit"])))
    total_units = int(csv["total_units_sold"])
    top_region = csv["top_region_by_revenue"]
    top_product = csv["top_product_by_revenue"]
    total_expenses = int(round(float(excel["total_q1_expenses"])))
    top_dept = excel["department_highest_expenses"]
    top_employee = excel["employee_highest_expenses"]

    checks_code = _make_automated_checks(csv, excel)

    return f"""---
id: {variant_id}
name: CSV and Excel Data Summarization ({variant_id})
category: data_analysis
grading_type: hybrid
timeout_seconds: 240
grading_weights:
  automated: 0.7
  llm_judge: 0.3
workspace_files:
  - source: {csv_asset}
    dest: quarterly_sales.csv
  - source: {xlsx_asset}
    dest: company_expenses.xlsx
---

## Prompt

{prompt}

---

## Expected Behavior

The agent should read both files, compute aggregate statistics, and write `data_summary.md`.

Key expected values:
- CSV total revenue: ${total_revenue:,}
- CSV total profit: ${total_profit:,}
- CSV total units: {total_units:,}
- CSV top region: {top_region}
- CSV top product: {top_product}
- Excel total Q1 expenses: ${total_expenses:,}
- Excel top department: {top_dept}
- Excel top employee: {top_employee}

---

## Grading Criteria

- [ ] Summary report file `data_summary.md` is created
- [ ] Total revenue is correctly reported (~${total_revenue:,})
- [ ] Total profit is correctly calculated (~${total_profit:,})
- [ ] Top region by revenue is identified ({top_region})
- [ ] Top product by revenue is identified ({top_product})
- [ ] Total Q1 expenses are correctly reported (~${total_expenses:,})
- [ ] Top spending department is identified ({top_dept})
- [ ] Top spending employee is identified ({top_employee})
- [ ] Budget vs actual comparison is included

---

## Automated Checks

```python
{checks_code}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="rl_train.jsonl path")
    parser.add_argument("--workspace-base", required=True, help="Directory containing workspaces/")
    parser.add_argument("--tasks-dir", default="tasks", help="Output tasks/ directory")
    parser.add_argument("--assets-dir", default="assets", help="Output assets/ directory")
    parser.add_argument("--prefix", default="rl_ss", help="Asset filename prefix")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    workspace_base = Path(args.workspace_base)
    tasks_dir = Path(args.tasks_dir)
    assets_dir = Path(args.assets_dir)

    rows = [json.loads(l) for l in input_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} variants from {input_path}")

    if not args.dry_run:
        tasks_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for row in rows:
        variant_id = row["variant_id"]
        src_ws = workspace_base / row["workspace_dir"]
        csv_src = src_ws / "quarterly_sales.csv"
        xlsx_src = src_ws / "company_expenses.xlsx"

        if not csv_src.exists() or not xlsx_src.exists():
            print(f"  SKIP {variant_id}: workspace files missing ({src_ws})")
            continue

        csv_asset = f"{args.prefix}_{variant_id}_quarterly_sales.csv"
        xlsx_asset = f"{args.prefix}_{variant_id}_company_expenses.xlsx"

        if not args.dry_run:
            shutil.copy2(csv_src, assets_dir / csv_asset)
            shutil.copy2(xlsx_src, assets_dir / xlsx_asset)

        md = _make_markdown(
            variant_id=variant_id,
            prompt=row["prompt"],
            csv=row["expected"]["csv"],
            excel=row["expected"]["excel"],
            csv_asset=csv_asset,
            xlsx_asset=xlsx_asset,
        )

        md_path = tasks_dir / f"{variant_id}.md"
        if not args.dry_run:
            md_path.write_text(md, encoding="utf-8")

        generated.append(variant_id)
        print(f"  OK {variant_id} -> {md_path.name}")

    print(f"\nGenerated {len(generated)} task markdowns")
    if args.dry_run:
        print("(dry-run, no files written)")


if __name__ == "__main__":
    main()
