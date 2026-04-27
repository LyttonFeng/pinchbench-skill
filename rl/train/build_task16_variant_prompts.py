#!/usr/bin/env python3
"""
Build a task_16 RL prompt pool from the canonical task plus mixed prompt variants.

Design goals:
- Keep the canonical task_id so workspace/grading stay unchanged.
- Reuse the old/base prompt pool rather than replacing it.
- Add targeted prompt groups derived from the failure taxonomy.
- Vary only the user-facing prompt wording and emphasis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_tasks import resolve_task_markdown_path  # noqa: E402
from rl.train.prepare_prompts import build_verl_row, parse_task_file  # noqa: E402


BASE_PROMPTS = [
    # canonical-like
    "Triage the emails in this inbox and write a markdown report to `triage_report.md` with priority, category, rationale, and recommended action for each item.",
    "Please process the inbox, prioritize each email, group related issues when appropriate, and produce the final output as `triage_report.md`.",
    # push completion
    "Read the inbox once, determine priorities and actions, and make sure you finish by writing a complete `triage_report.md` file.",
    "Your goal is to deliver a usable `triage_report.md`, not just read emails. Review the inbox, prioritize each thread, and write the report.",
    # push structure
    "Create a structured triage report in `triage_report.md` covering every inbox email with: priority, category, summary, and next action.",
    "Produce `triage_report.md` with one structured entry per email/thread, including priority, category, rationale, and follow-up action.",
    # push incident linkage
    "Triage this inbox with special attention to related operational incidents. If multiple emails refer to the same outage/alert, link them in `triage_report.md`.",
    "As you triage the inbox, connect emails that belong to the same incident and capture that linkage explicitly in `triage_report.md`.",
    # push business weighting
    "Prioritize the inbox with strong business judgment. Important customer, outage, and security items should clearly outrank routine internal or newsletter mail.",
    "When triaging, weight production outages, security tasks, and high-value customer threads appropriately. Reflect those priorities in `triage_report.md`.",
    # push efficiency / avoid rereads
    "Do an efficient inbox triage pass and avoid unnecessary rereading. Then write the final result to `triage_report.md`.",
    "Make one effective pass through the inbox, extract what matters, and finish with a clear `triage_report.md`.",
    # emphasize complete coverage
    "Cover the entire inbox in `triage_report.md`. Every email should be accounted for, even if some are low priority.",
    "Write `triage_report.md` so that no inbox item is left untriaged. Include low-priority items briefly, but do not omit them.",
    # emphasize concise but complete execution
    "Triage the inbox efficiently, avoid looping over the same messages, and produce a complete `triage_report.md` at the end.",
    "Use a single efficient workflow: inspect the inbox, determine priorities, then write the final `triage_report.md` without redundant rereads.",
    # emphasize outage/security business logic
    "Treat outage-related, production-impacting, and security-sensitive emails as high-priority unless there is strong evidence otherwise, and reflect that in `triage_report.md`.",
    "Bias your triage toward operational urgency: production incidents, security actions, and release-blocking issues should rank above routine requests.",
    # emphasize customer impact
    "Pay attention to customer impact. High-value customer emails should be surfaced clearly and prioritized appropriately in `triage_report.md`.",
    "Use business impact in your ranking: customer-facing disruptions and important customer threads should stand out in the final report.",
    # emphasize linkage and deduplication
    "If multiple emails describe the same underlying problem, avoid treating them as isolated items. Link them clearly in `triage_report.md`.",
    "Deduplicate related threads mentally while triaging. The final report should show when two emails belong to the same incident.",
    # emphasize output schema
    "The final `triage_report.md` should be easy for an operator to use: priority, category, rationale, and next action should be explicit for each item.",
    "Write `triage_report.md` as an operator-ready triage document with explicit priority, summary, rationale, and recommended action.",
    # emphasize avoiding failure mode of no report
    "Do not stop after reading the inbox. The task is only complete once `triage_report.md` has been written.",
    "Reading emails is not enough. Finish the job by writing a complete `triage_report.md`.",
    # emphasize high-signal grouping
    "Look for the highest-signal items first: production outages, security work, major customer impact, and release blockers. Then write `triage_report.md`.",
    "Focus triage on operationally important signals, then capture the whole inbox in `triage_report.md` with clear prioritization.",
    # emphasize summary quality
    "End with a report that a teammate could act on immediately. `triage_report.md` should be structured, actionable, and complete.",
    "Produce a `triage_report.md` that is concise, complete, and actionable for a human operator taking over the inbox.",
]

TARGETED_PROMPT_GROUPS: dict[str, list[str]] = {
    "email13_coverage": [
        "Do not omit low-salience but operationally important alerts. Make sure every inbox item, including correlated monitoring alerts, appears in `triage_report.md`.",
        "When writing `triage_report.md`, account for every email explicitly. A monitoring alert that relates to an active outage must not be skipped.",
        "Cover the whole inbox. If a message looks like a supporting alert for a production issue, include it explicitly instead of treating it as disposable noise.",
        "Make sure `triage_report.md` includes all emails, including alerts that may look minor on their own but matter in operational context.",
    ],
    "incident_linkage": [
        "If an outage email and a monitoring alert refer to the same operational problem, connect them explicitly in `triage_report.md` rather than listing them as unrelated items.",
        "Related outage and alert emails should be grouped into one incident view in `triage_report.md` whenever the evidence supports it.",
        "While triaging, look for emails that belong to the same incident and make that linkage explicit in the report.",
        "Do not treat correlated outage and alert threads as isolated events. Show incident linkage clearly in `triage_report.md`.",
    ],
    "email13_priority": [
        "A correlated latency alert during an ongoing outage should rank near the top, not with routine low-priority mail. Reflect that in `triage_report.md`.",
        "Do not downgrade a monitoring alert that is tied to an active production incident. Prioritize it appropriately in the final report.",
        "When an alert appears to confirm or extend an outage, rank it as an operationally important item rather than background noise.",
        "Use operational context when assigning priority: a correlated alert during an incident should not be buried in the queue.",
    ],
    "bigclient_weighting": [
        "High-value customer threads should stand out clearly in `triage_report.md`. Customer impact and revenue risk must affect priority.",
        "Do not treat a major customer thread like routine mail. Surface high-value customer impact explicitly in the final triage.",
        "When business impact is high, rank customer-facing issues above routine internal requests and administrative threads.",
        "Use revenue and customer-risk weighting in your prioritization. Important customer emails must be clearly surfaced in `triage_report.md`.",
    ],
    "security_weighting": [
        "Security and compliance deadlines require elevated priority and a concrete next action in `triage_report.md`.",
        "Do not file security/compliance work as ordinary admin mail. Give it explicit priority and follow-up action.",
        "When a message carries security or compliance urgency, reflect that with a higher rank and a specific recommended action.",
        "Security-sensitive or compliance-related items should be clearly elevated above routine inbox noise in the final report.",
    ],
    "closure_and_stop_reread": [
        "Do one effective inbox pass, avoid looping over the same messages, and switch to writing `triage_report.md` once coverage is sufficient.",
        "Do not keep rereading the full inbox after you already have enough evidence. Finish by writing a complete `triage_report.md`.",
        "The task is to produce a finished triage artifact, not to endlessly re-open the inbox. Stop rereading and write the report.",
        "Once you have covered the inbox, stop gathering and deliver a complete `triage_report.md` with clear priorities and actions.",
    ],
    "incident_graph": [
        "Before assigning final priorities, first build an incident graph for the inbox: determine which emails belong to the same operational event, which are standalone work items, and which are low-value noise. Then write `triage_report.md` from that global view.",
        "Do not triage each email in isolation. First identify incident groups across the inbox, then assign priorities from the incident level down to the email level in `triage_report.md`.",
        "Use a two-stage workflow: (1) build an inbox-level event map, grouping related outage/alert emails together, and (2) write `triage_report.md` from that event map rather than from isolated local judgments.",
        "Model the inbox as incidents plus standalone tasks. Group related emails first, then write the final report using that global grouping.",
    ],
    "priority_propagation": [
        "After identifying related emails, propagate priority from the incident level to all member emails. If one email establishes a P0 outage, related alert emails should not be ranked as routine low-priority items.",
        "Use incident-level priority propagation: when two emails belong to the same critical incident, their priorities must stay consistent with that shared incident severity.",
        "Do not let a correlated alert inherit a lower priority just because its wording looks less urgent. Incident membership should affect priority assignment in `triage_report.md`.",
        "Assign priority globally, not locally: critical incident context should raise the ranking of supporting alert emails and related follow-up threads.",
    ],
    "report_schema_incident_groups": [
        "Write `triage_report.md` with two top-level sections: `## Incident Groups` and `## Standalone Items`. Under `Incident Groups`, list each incident with member emails, shared priority, rationale, and action. Under `Standalone Items`, cover the remaining emails.",
        "Structure the report explicitly around grouped incidents. Use one section for grouped operational incidents and another for standalone items, so the global inbox state is visible.",
        "The final report must expose the inbox-level structure. Use `## Incident Groups` for linked outage/alert threads and `## Standalone Items` for everything else.",
        "Do not output only a flat per-email list. Make the report show incident grouping first, then standalone tasks, so priority decisions are traceable.",
    ],
    "report_schema_priority_fields": [
        "In `triage_report.md`, every incident group or standalone item must include: covered emails, priority, category, rationale, and recommended action.",
        "Use an operator-readable schema in the report: each grouped incident or standalone item should show member emails, priority, category, why it matters, and what to do next.",
        "Make the report schema explicit. For each incident group or standalone item, include the emails covered, the shared priority, the category, the reasoning, and the next action.",
        "The report should make global judgment legible: show which emails belong together and give each group or standalone item a clear priority, rationale, and action.",
    ],
}


TASK16_REWARD_RUBRIC = {
    "required_report_schema": "incident_groups_v1",
    "expected_incident_groups": [
        {
            "id": "production_database_incident",
            "emails": ["email_01", "email_13"],
            "priority": "p0",
        }
    ],
    "expected_priorities": {
        "email_01": "p0",
        "email_13": "p0",
        "email_05": "p1",
        "email_08": "p1",
    },
    "minimum_email_coverage": 10,
}


def _row_with_group(task: dict, repeat_idx: int, group: str) -> dict:
    row = build_verl_row(task, repeat_idx=repeat_idx)
    row["extra_info"]["prompt_group"] = group
    row["extra_info"]["reward_rubric"] = TASK16_REWARD_RUBRIC
    return row


def _select_targeted_val_entries(
    prompt_entries: list[tuple[str, str]],
    base_val_count: int,
) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []

    # Keep a small amount of plain/base validation so val does not become only schema-driven.
    base_seen = 0
    for group, prompt in prompt_entries:
        if group == "base" and base_seen < max(0, base_val_count):
            selected.append((group, prompt))
            base_seen += 1

    # Cover the newer targeted capabilities explicitly.
    targeted_order = [
        "incident_linkage",
        "email13_priority",
        "bigclient_weighting",
        "security_weighting",
        "incident_graph",
        "priority_propagation",
        "report_schema_incident_groups",
        "report_schema_priority_fields",
    ]
    for wanted_group in targeted_order:
        for group, prompt in prompt_entries:
            if group == wanted_group:
                selected.append((group, prompt))
                break

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task_16 RL prompt variants")
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rl/data/prompts_task16_variants"),
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=10,
        help="Deprecated compatibility arg. Targeted val selection is used instead of taking the first N variants.",
    )
    args = parser.parse_args()

    task_path = resolve_task_markdown_path(args.tasks_dir, "task_16_email_triage")
    if not task_path.exists():
        raise SystemExit(f"Task file not found: {task_path}")

    canonical = parse_task_file(task_path)
    rows = [_row_with_group(canonical, repeat_idx=0, group="canonical")]

    variant_tasks: list[dict] = []
    prompt_entries: list[tuple[str, str]] = []
    prompt_entries.extend(("base", p) for p in BASE_PROMPTS)
    for group, prompts in TARGETED_PROMPT_GROUPS.items():
        prompt_entries.extend((group, p) for p in prompts)

    for i, (group, prompt) in enumerate(prompt_entries, start=1):
        task = dict(canonical)
        task["prompt"] = prompt
        variant_tasks.append(task)
        rows.append(_row_with_group(task, repeat_idx=i, group=group))

    val_entries = [("canonical", canonical["prompt"])] + _select_targeted_val_entries(
        prompt_entries,
        base_val_count=2,
    )
    val_rows = []
    for i, (group, prompt) in enumerate(val_entries):
        task = dict(canonical)
        task["prompt"] = prompt
        val_rows.append(_row_with_group(task, repeat_idx=999 + i, group=group))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "val.parquet"
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    print(f"Wrote {len(rows)} train prompts to {train_path}")
    print(f"Wrote {len(val_rows)} val prompts to {val_path}")
    print(f"Canonical task_id: {canonical['task_id']}")
    print(f"Variant prompts: {len(variant_tasks)}")
    print(f"Base prompts: {len(BASE_PROMPTS)}")
    print(f"Targeted prompts: {sum(len(v) for v in TARGETED_PROMPT_GROUPS.values())}")
    for group, prompts in TARGETED_PROMPT_GROUPS.items():
        print(f"  group[{group}] = {len(prompts)}")
    preview_prompts = [canonical["prompt"]] + [p for _, p in prompt_entries[:4]]
    for idx, prompt in enumerate(preview_prompts):
        preview = " ".join(prompt.split())[:120]
        print(f"  [{idx}] {preview}...")


if __name__ == "__main__":
    main()
