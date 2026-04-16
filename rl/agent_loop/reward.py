"""
Process reward + terminal reward computation for PinchBench RL.

Four ablation modes:
  Mode A (baseline):    per-turn reward = 0, last turn gets terminal_reward
  Mode B (rule-only):   per-turn reward from generic behavior rules (fast, no LLM)
  Mode C (self-judge):  Qwen3-4B self-judge with rubric in prompt, then terminal reward
  Mode D (oracle-judge): qwen-plus judge with rubric in prompt, then terminal reward

Default mode: C (self-judge) — 自进化：模型自己评判自己

Terminal reward: {-1, +1}  (task fail / succeed)
Process reward:  [-0.5, +0.2] per turn

PRM (self-judge) uses task goal + optional_hints (non-binding) + common_mistakes.
reference_steps stay in the rubric as guidance inside the prompt, but do not feed into the
numeric reward for self-judge/oracle-judge.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    logger.addHandler(_h)


# ══════════════════════════════════════════════════════════════
#  Per-task PRM rubrics (goal + optional_hints for PRM; reference_steps for Mode B rules)
# ══════════════════════════════════════════════════════════════

_TASK_OPTIONAL_HINTS_DEFAULT = (
    "Typical agents gather context, take actions, and verify. When the task requires reading files, searching, "
    "editing, or checking facts, prefer using a tool first and only then summarize the result. Many valid tool "
    "orders and strategies can satisfy the task goal."
)


def _optional_hints_for_prm(rubric: dict[str, Any]) -> str:
    """Short non-binding hints for self-judge PRM only (Mode B still uses reference_steps)."""
    h = rubric.get("optional_hints")
    if isinstance(h, str) and h.strip():
        return h.strip()
    return _TASK_OPTIONAL_HINTS_DEFAULT


TASK_RUBRICS: dict[str, dict[str, Any]] = {
    "task_02_stock": {
        "goal": "Research Apple (AAPL) stock price, save to stock_report.txt including the literal ticker AAPL, price, date, and market summary (PinchBench automated grading requires the substring AAPL in the file).",
        "optional_hints": (
            "Useful signals: the agent gathers fresh AAPL market data with a tool first, then writes a grounded "
            "stock_report.txt with ticker, price, date, and a concise summary. "
            "If web search returns incomplete or stale price data, still write stock_report.txt with the best "
            "available figure and a caveat noting the data limitation — writing an approximate report always beats "
            "writing nothing."
        ),
        "reference_steps": [
            "Gather current AAPL market data from a reliable source",
            "If needed, fetch more detail or confirm the quote from another source",
            "Write stock_report.txt with AAPL, price, date, and a market summary",
            "Verify the file exists and contains the required ticker token",
        ],
        "common_mistakes": [
            "Refusing to create stock_report.txt with a message like 'cannot retrieve real-time prices due to privacy constraints' — this is the worst outcome; the grader requires the file to exist; use whatever search results you have, note limitations, and write the file",
            "Answering from memory or text-only before checking current AAPL data",
            "Writing only 'Apple' or company name without the literal token AAPL (automated grader checks \\bAAPL\\b)",
            "Searching but never writing the file (premature termination)",
            "Writing the file without any web_search first (fabricating data)",
            "Repeating the same failed search query without changing it",
        ],
        "qwen_plus_stats": "7 turns, 6 tool calls, 745 bytes output",
    },
    "task_12_skill_search": {
        "goal": "Update config/settings.json and config/database.yml per prompt: localhost→prod-db.example.com, myapp_dev/myapp_test→myapp_prod, log debug→warn, API URL to https://api.example.com. Automated grader checks file contents.",
        "optional_hints": (
            "Useful signals: the agent inspects both config files with tools first, applies every required substitution, "
            "then verifies the final contents before finishing."
        ),
        "reference_steps": [
            "Inspect both config files before editing",
            "Apply every required replacement across JSON and YAML",
            "Check that no required field or value was missed",
            "Verify the final file contents after editing",
        ],
        "common_mistakes": [
            "Calling read('config/*') or any glob/wildcard path — the read tool requires exact filenames; instead call read on each file directly by name (config/settings.json and config/database.yml are given in the task prompt)",
            "Giving up and asking the user to confirm the path after a single ENOENT error — when the files are listed in the task prompt, use those exact paths directly without wildcards",
            "Skipping inspection and jumping straight to edit without first reading both files to understand their exact structure",
            "Repeating the same failing edit call (same oldText) without adding more surrounding context to disambiguate duplicate field names — database.yml has 'host: localhost' in both development and test blocks, so the oldText must include enough surrounding lines to be unique",
            "Not switching strategy after an edit fails: if 'Found N occurrences' error appears, the fix is to include the parent block header (e.g. 'development:\\n  host: localhost') in oldText, not to retry the same short oldText",
            "Missing one file or one replacement class (localhost, DB name, log level, API URL)",
            "Not verifying with read after edits to confirm all replacements landed correctly",
        ],
        "qwen_plus_stats": "9 turns, 8 tool calls. Used edit with expanded context for ambiguous matches.",
    },
    "task_10_workflow": {
        "goal": "Read workspace config.json, extract API endpoint, write a Python script that calls it, document in NOTES.md (same as PinchBench task; hybrid grader checks script + notes quality).",
        "optional_hints": (
            "Useful signals: the agent reads config.json first, derives the endpoint with a tool, writes a runnable "
            "script that uses that endpoint, and leaves concise notes that match the actual implementation. The script "
            "should be grounded in runtime config access, not merely a copied value. The strongest version reads or "
            "derives the endpoint at runtime from config.json instead of copying the sample URL."
        ),
        "reference_steps": [
            "Use config.json as the source of truth for the endpoint",
            "Produce a script that is grounded in the config-derived API details",
            "Keep the script runnable and avoid hardcoding the URL",
            "Write NOTES.md that accurately matches what the script does",
        ],
        "common_mistakes": [
            "Directly answering or drafting code before reading config.json",
            "Reading config.json but still hardcoding the endpoint in the script",
            "Treating the sample URL as the only acceptable URL instead of preserving runtime config loading",
            "Script that does not actually reflect the config-derived endpoint",
            "Missing NOTES.md or notes that do not match the script",
            "Producing a script that only works for the sample config instead of the general runtime pattern",
        ],
        "qwen_plus_stats": "4 turns, 3 tool calls. Python: 1704 bytes, NOTES: 1389 bytes.",
    },
    "task_22_second_brain": {
        "goal": "Multi-session task (OpenClaw sessions in frontmatter): Session 1 — write user facts to memory/MEMORY.md (Rust, Jan 15 2024, Dr. Elena Vasquez, NeonDB, secret phrase). Session 2 — answer language + project from file. Session 3 (new session) — read MEMORY.md and answer all 5 recall questions. Matches PinchBench hybrid grader.",
        "optional_hints": (
            "Useful signals: the agent persists the required facts correctly with a tool, retrieves them from MEMORY.md "
            "in a fresh session, and answers every recall item consistently from saved memory."
        ),
        "reference_steps": [
            "Persist the required facts into memory/MEMORY.md without missing or corrupting them",
            "Use the saved memory rather than guessing when answering follow-up questions",
            "Confirm the facts are still available in a fresh session by reading MEMORY.md again",
            "Answer all five recall questions exactly from the stored facts",
        ],
        "common_mistakes": [
            "Trying to answer recall questions without re-reading MEMORY.md in the new session",
            "Wrong path (must be memory/MEMORY.md under workspace)",
            "Missing any of the required facts in MEMORY.md",
            "Session 3 not reading the file and hallucinating answers",
            "Omitting one of the five recall items or answering inconsistently across sessions",
        ],
        "qwen_plus_stats": "Multi-session; ~3 OpenClaw session invocations per benchmark run.",
    },
    "task_16_email_triage": {
        "goal": "Read all 13 files in inbox/ (email_01.txt … email_13.txt), then write triage_report.md: top summary + day plan; for each email assign Priority P0–P4, Category (incident/client/internal-request/administrative/code-review/automated/newsletter/spam), and recommended action; sort entries by priority (most urgent first). PinchBench checks: production outage email as P0; monitoring alert tied to same incident; BigClient email P0/P1; promotional/spam email P4; report structure and summary section.",
        "optional_hints": (
            "Useful signals: the agent covers the whole inbox with read tools before writing, correctly flags the urgent "
            "incident and big-client mail, and produces a report whose priorities match the content."
        ),
        "reference_steps": [
            "Cover the entire inbox before committing to the report",
            "Correctly identify the highest-priority incident and important client mail",
            "Separate low-value spam/newsletter mail from actionable items",
            "Write triage_report.md with priorities, categories, and actions that match the email content",
        ],
        "common_mistakes": [
            "Starting the report from memory or after only a partial inbox read",
            "Writing triage_report.md before reading all 13 emails or omitting any email from the report",
            "Missing file triage_report.md or wrong filename",
            "Production database outage / CTO war-room email not labeled P0",
            "API latency monitoring alert (email 13) not linked or grouped with the DB outage incident",
            "BigClient / $2M contract email not P0 or P1",
            "Promotional flash-sale / spam-like email not P4 or lowest tier",
            "Report not sorted by priority (most urgent first) or missing the executive summary/day plan",
            "Missing per-email recommended action or category keywords the grader expects",
        ],
        "qwen_plus_stats": "15 turns, 14 tool calls; read each inbox file; triage_report.md with summary then priority-sorted entries.",
    },
    "task_18_spreadsheet_summary": {
        "goal": "Analyze workspace CSV + XLSX per task, compute real aggregates, write data_summary.md (PinchBench id=task_18_spreadsheet_summary; markdown file may be task_19_spreadsheet_summary.md on disk).",
        "optional_hints": (
            "Useful signals: the agent reads the CSV and computes real aggregates from the rows. For xlsx: the read "
            "tool returns binary for .xlsx files — this is expected. Use the column names and row count described in "
            "the task prompt as structural guidance, then write data_summary.md with real CSV numbers and "
            "clearly-estimated xlsx figures from the prompt description. Writing a partial report that accurately "
            "handles the CSV is far better than writing nothing."
        ),
        "reference_steps": [
            "Extract real numeric data from the CSV and XLSX sources",
            "Compute the aggregates with a verifiable method",
            "Write data_summary.md using the computed values",
            "Check that the reported numbers match the extracted data",
        ],
        "common_mistakes": [
            "Attempting to summarize the spreadsheet without extracting numeric data first",
            "Pretending to read .xlsx as UTF-8 text and inventing numbers",
            "Using only one file and ignoring the other source when both are required",
            "Writing summary without any successful numeric extraction",
            "Entering a repetitive thinking loop when xlsx returns binary — stop looping; use the xlsx structure described in the task prompt (sheet names, column names, row count) to write estimated xlsx figures in the report",
            "Writing nothing because xlsx cannot be parsed as text — data_summary.md must exist; accurate CSV analysis plus xlsx-prompt-based estimates is a valid and scorable report",
            "Reading xlsx with the read tool and then pretending the binary garbage contains actual numbers — acknowledge the limitation and use prompt-provided structure instead",
            "Reporting aggregates that cannot be traced back to the raw rows",
        ],
        "qwen_plus_stats": "7 turns, 6 tool calls, 1728 bytes. Switched from pandas to awk when needed.",
    },
    "task_18_market_research": {
        "goal": "Write market_research.md: enterprise observability/APM landscape, top ~5 players, differentiators, pricing models, trends; comparison table and analyst-style structure. Use web search if available; else knowledge (per task text).",
        "optional_hints": (
            "Useful signals: the agent identifies major players using research tools first when available, compares pricing "
            "and differentiation, and writes a structured analyst-style brief instead of generic commentary. The brief "
            "should name concrete vendors and include at least one specific pricing or product signal. The comparison "
            "table should compare the same dimensions across vendors."
        ),
        "reference_steps": [
            "Collect current market context on observability/APM",
            "Compare several meaningful competitors and their positioning",
            "Include pricing, differentiators, and trend signals",
            "Write a structured market_research.md with an executive summary and comparison table",
        ],
        "common_mistakes": [
            "Writing generic commentary without first collecting market context when web/search tools are available",
            "Missing market_research.md or fewer than 5 meaningful competitor sections",
            "Generic fluff with no pricing or trends",
            "Naming vendors without concrete differentiators or product/pricing evidence",
            "A comparison table where rows/columns are not aligned across vendors",
            "Search queries with stale years when using web_search",
            "Writing a table that does not actually compare the same dimensions across vendors",
        ],
        "qwen_plus_stats": "5 turns, 4 tool calls, 8412 bytes. Three search angles: leaders, comparison, pricing.",
    },
    "task_24_polymarket_briefing": {
        "goal": "Write polymarket_briefing.md: top 3 trending Polymarket markets (real/active), Yes/No odds, related news within 48h. Task allows gamma API or polymarket.com / web search; do not fabricate markets or odds.",
        "optional_hints": (
            "Useful signals: the agent finds real active markets with tools first, checks current odds, and links each "
            "market to fresh supporting news without inventing details. Prefer the gamma API for market discovery; if "
            "one source fails, try the permitted alternative before giving up. The gamma API should be treated as the "
            "primary discovery path, with web search as fallback for corroboration/news."
        ),
        "reference_steps": [
            "Find real active Polymarket markets and current odds",
            "Corroborate each market with recent news",
            "Write a briefing that covers the top three markets with dates and context",
            "Avoid inventing market names, odds, or sources",
        ],
        "common_mistakes": [
            "Writing the wrong year/date in the report header (e.g. '2023-10-25') — the header must show today's actual date; never use dates from training memory",
            "Falling back to 2023-era Polymarket training knowledge when web search returns poor results — 2023 markets (FTX, Iran ceasefire from that era, Bitcoin $50K Dec 2023) are stale and will fail grading; do additional targeted searches instead",
            "Stopping after 2 search attempts when both returned poor results — try at least 3–4 different queries (e.g. 'polymarket trending 2026', 'gamma.io markets', specific topic searches) before writing",
            "Guessing markets or odds before checking a source",
            "Hallucinated market names or odds",
            "News not tied to the market or not recent",
            "Wrong filename or missing 3 market blocks",
            "Giving up after a failed search instead of trying the gamma API / alternate permitted source",
            "Using web search as the primary discovery path when the gamma API is available",
            "Reporting a market as 'unavailable' when the allowed fallback path is still available",
        ],
        "qwen_plus_stats": "8 turns, 7 tool calls, 1719 bytes. Searched trends then each topic individually.",
    },
}

# Generic rubric for unknown tasks
_GENERIC_RUBRIC = {
    "goal": "Complete the assigned task using appropriate tools.",
    "optional_hints": _TASK_OPTIONAL_HINTS_DEFAULT,
    "reference_steps": [
        "1. Gather information (read files, search web)",
        "2. Process/analyze the information",
        "3. Take action (write files, execute commands)",
        "4. Verify the results",
    ],
    "common_mistakes": [
        "Skipping information gathering and going straight to action",
        "Not verifying results after taking action",
        "Repeating failed commands without changing strategy",
        "Empty response without any tool call",
    ],
}

# PinchBench frontmatter `id` is task_18_spreadsheet_summary; older code/logs used task_19_*.
_TASK_ID_ALIASES: dict[str, str] = {
    "task_19_spreadsheet_summary": "task_18_spreadsheet_summary",
}


def _canonical_task_id_for_rubric(task_id: str) -> str:
    return _TASK_ID_ALIASES.get(task_id, task_id)


def _get_task_rubric(task_id: str) -> dict[str, Any]:
    return TASK_RUBRICS.get(_canonical_task_id_for_rubric(task_id), _GENERIC_RUBRIC)


TERMINAL_REWARD_WEIGHT = float(
    os.environ.get("PINCHBENCH_TERMINAL_REWARD_WEIGHT", "0.3")
)


def _resolve_model_id(candidates: list[str], desired_model: str) -> str:
    """Pick the best served model ID for a requested judge model name.

    We accept exact matches first, then basename matches so that
    `Qwen3-4B` can resolve to `Qwen/Qwen3-4B`.
    """
    if not candidates:
        return desired_model

    if desired_model in candidates:
        return desired_model

    desired_base = desired_model.split("/")[-1]
    basename_matches = [
        model_id for model_id in candidates if model_id.split("/")[-1] == desired_base
    ]
    if basename_matches:
        return basename_matches[0]

    partial_matches = [
        model_id for model_id in candidates
        if desired_model in model_id or model_id in desired_model
    ]
    if partial_matches:
        return partial_matches[0]

    return candidates[0]


async def _resolve_judge_model_async(vllm_base_url: str, desired_model: str) -> str:
    """Resolve the actual model name exposed by the current vLLM server."""
    import aiohttp

    endpoint = f"{vllm_base_url.rstrip('/')}/models"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                endpoint,
                headers={"Authorization": "Bearer dummy"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Judge model lookup failed %s -> HTTP %s", endpoint, resp.status)
                    return desired_model
                payload = await resp.json()
    except Exception as exc:
        logger.warning("Judge model lookup failed %s: %s", endpoint, exc)
        return desired_model

    served_models = [
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    resolved = _resolve_model_id(served_models, desired_model)
    if resolved != desired_model:
        logger.info("Resolved judge model %s -> %s from %s", desired_model, resolved, endpoint)
    return resolved


def _resolve_judge_model_sync(vllm_base_url: str, desired_model: str) -> str:
    """Sync version used by the fallback judge path."""
    from urllib import request as urllib_request

    endpoint = f"{vllm_base_url.rstrip('/')}/models"
    try:
        req = urllib_request.Request(endpoint, method="GET")
        with urllib_request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Judge model lookup failed %s: %s", endpoint, exc)
        return desired_model

    served_models = [
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    resolved = _resolve_model_id(served_models, desired_model)
    if resolved != desired_model:
        logger.info("Resolved judge model %s -> %s from %s", desired_model, resolved, endpoint)
    return resolved


# ══════════════════════════════════════════════════════════════
#  PRM Prompt Construction
# ══════════════════════════════════════════════════════════════

def build_prm_prompt(
    task_id: str,
    task_prompt: str,
    turn_index: int,
    total_turns: int,
    current_turn: dict,
    prev_turns: list[dict],
    tool_result: Optional[str] = None,
    mode: str = "self-judge",
) -> str:
    """Build the PRM scoring prompt for one assistant turn.

    The prompt gives the judge:
      - Task goal
      - Optional hints (illustrative; not a mandatory script)
      - Common mistakes to watch for
      - The agent's previous actions (context)
      - The current turn to evaluate
    """
    rubric = _get_task_rubric(task_id)

    # Format current turn info
    tool_name = _get_tool_name(current_turn) or "(no tool call)"
    tool_args = _get_tool_args(current_turn)
    content = current_turn.get("content", "")

    # Truncate long content
    content_preview = content[:500] + "..." if len(content) > 500 else content
    tool_result_preview = ""
    if tool_result:
        tool_result_preview = tool_result[:500] + "..." if len(tool_result) > 500 else tool_result

    # Format previous actions summary
    prev_summary_lines = []
    for i, t in enumerate(prev_turns):
        if t.get("role") == "assistant":
            t_tool = _get_tool_name(t) or "text-only"
            t_args = _get_tool_args(t)
            arg_str = ""
            if t_args:
                arg_preview = str(t_args)[:100]
                arg_str = f"({arg_preview})"
            prev_summary_lines.append(f"  Turn {i+1}: {t_tool}{arg_str}")
        elif t.get("role") == "tool":
            tc = t.get("content", "")
            status = "ERROR" if _is_error_result(tc) else "OK"
            prev_summary_lines.append(f"    → Result: {status}")

    prev_summary = "\n".join(prev_summary_lines[-20:]) if prev_summary_lines else "  (first turn)"

    optional_hints = _optional_hints_for_prm(rubric)

    # Build common mistakes string
    mistakes = "\n".join(f"- {m}" for m in rubric.get("common_mistakes", []))

    # Format tool args for display
    args_display = json.dumps(tool_args, ensure_ascii=False, indent=2)[:300] if tool_args else "(none)"

    prompt = f"""You are an AI Agent behavior evaluator. Score whether the current step helps the agent make real progress on the task.

## Task Goal
{rubric.get("goal", task_prompt[:500])}

## Tool-Use Preference
When the task requires reading files, searching, editing, computing, or checking facts, prefer a tool-first step over a text-only answer.
Direct text-only summaries are low value if the agent has not yet gathered evidence with a tool.

## Optional Hints (illustrative only — many valid approaches exist)
{optional_hints}

## Common Mistakes to Watch For
{mistakes}

## Agent's Previous Actions (Turn 1 to {turn_index})
{prev_summary}

## Current Turn ({turn_index + 1} of ~{total_turns} expected)
Tool: {tool_name}
Arguments: {args_display}
Agent's text: {content_preview}"""

    if tool_result_preview:
        prompt += f"\nTool result: {tool_result_preview}"

    prompt += """

## Scoring Instructions
Evaluate whether this step makes the agent more likely to complete the task successfully.
Consider:
- Is the agent using a tool to gather evidence or take action when that is needed?
- Does this step gather useful information, take a correct action, or reduce uncertainty?
- Does this step avoid the common mistakes listed above?
- Does this step contribute to completing the task, even if the exact path differs from the hints?

Score range: -0.5 (harmful/wasteful step) to +0.2 (excellent progress)
- +0.1 to +0.2: Step clearly advances toward the goal (e.g., correct tool with good arguments)
- +0.05 to +0.15: Reasonable step, some progress
- 0.0: Neutral, neither helpful nor harmful
- -0.1 to -0.2: Suboptimal but not terrible (e.g., redundant action)
- -0.3 to -0.5: Actively harmful (e.g., fabricating data, repeating failed commands)

Respond with ONLY a JSON object, no other text:
{"score": <float between -0.5 and 0.2>, "reason": "<one sentence explanation>"}"""

    return prompt


# ══════════════════════════════════════════════════════════════
#  LLM Judge call
# ══════════════════════════════════════════════════════════════

async def call_llm_judge(
    prompt: str,
    vllm_base_url: str = "http://localhost:9090/v1",
    model: str = "Qwen3-4B",
    api_key: str = "dummy",
    timeout: float = 30.0,
) -> float:
    """Call LLM (Qwen3-4B via vLLM) to score a single turn.

    Uses /v1/completions (text completion) because veRL's vLLMHttpServer
    does not expose /v1/chat/completions.

    Returns score in [-0.5, +0.2]. Falls back to 0.0 on any error.
    """
    import aiohttp

    full_prompt = (
        "<|im_start|>system\nYou are a strict scoring function. "
        "Respond with ONLY a JSON object. No thinking, no explanation.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    )

    payload = {
        "model": model,
        "prompt": full_prompt,
        "temperature": 0.1,
        "max_tokens": 64,
        "stop": ["<|im_end|>"],
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{vllm_base_url}/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("PRM judge HTTP %d: %s", resp.status, body[:200])
                    return 0.0
                data = await resp.json()
    except Exception as e:
        logger.warning("PRM judge request failed: %s", e)
        return 0.0

    try:
        text = data["choices"][0]["text"].strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        result = json.loads(text)
        score = float(result.get("score", 0.0))
        reason = result.get("reason", "")
        print(f"[PRM] Judge scored: {score:.2f}, reason: {reason}")
        return max(-0.5, min(0.2, score))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning("PRM judge parse failed: %s, raw: %s", e, text[:200] if 'text' in dir() else "N/A")
        return 0.0


def call_llm_judge_sync(
    prompt: str,
    vllm_base_url: str = "http://localhost:9090/v1",
    model: str = "Qwen3-4B",
    api_key: str = "dummy",
    timeout: float = 30.0,
) -> float:
    """Synchronous wrapper for call_llm_judge (uses /v1/completions)."""
    from urllib import request as urllib_request

    full_prompt = (
        "<|im_start|>system\nYou are a strict scoring function. "
        "Respond with ONLY a JSON object. No thinking, no explanation.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    )

    payload = json.dumps({
        "model": model,
        "prompt": full_prompt,
        "temperature": 0.1,
        "max_tokens": 64,
        "stop": ["<|im_end|>"],
    }).encode("utf-8")

    req = urllib_request.Request(
        f"{vllm_base_url}/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning("PRM judge sync request failed: %s", e)
        return 0.0

    try:
        text = data["choices"][0]["text"].strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        result = json.loads(text)
        score = float(result.get("score", 0.0))
        return max(-0.5, min(0.2, score))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning("PRM judge sync parse failed: %s", e)
        return 0.0


# ══════════════════════════════════════════════════════════════
#  Helper functions (shared with rule-based mode)
# ══════════════════════════════════════════════════════════════

def _extract_tool_calls(turn: dict) -> list[dict]:
    raw = turn.get("tool_calls", [])
    calls = []
    for tc in raw:
        if isinstance(tc, dict):
            func = tc.get("function", tc)
            calls.append({
                "name": func.get("name", tc.get("name", "")),
                "arguments": func.get("arguments", tc.get("arguments", {})),
            })
    return calls


def _get_tool_name(turn: dict) -> Optional[str]:
    calls = _extract_tool_calls(turn)
    return calls[0]["name"] if calls else None


def _get_tool_args(turn: dict) -> dict:
    calls = _extract_tool_calls(turn)
    if not calls:
        return {}
    args = calls[0].get("arguments", {})
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {"_raw": args}
    return args


def _is_error_result(content: str) -> bool:
    error_patterns = [
        r"Traceback \(most recent call last\)",
        r"(?:Error|Exception):",
        r"command not found",
        r"No such file or directory",
        r"Permission denied",
        r"ModuleNotFoundError",
    ]
    for p in error_patterns:
        if re.search(p, content):
            return True
    return False


# ══════════════════════════════════════════════════════════════
#  Rule-based reward (Mode B fallback)
# ══════════════════════════════════════════════════════════════

def _parse_rubric_expected_tools(rubric: dict) -> list[str]:
    """Extract expected tool types from reference_steps in order."""
    tool_keywords = {
        "web_search": ["web_search", "search"],
        "web_fetch": ["web_fetch", "fetch"],
        "read": ["read", "examine", "read each"],
        "write": ["write", "create", "save"],
        "edit": ["edit", "replace", "modify"],
        "exec": ["exec", "run", "awk", "python", "pandas", "compute"],
    }
    steps = rubric.get("reference_steps", [])
    ordered_tools = []
    for step in steps:
        step_lower = step.lower()
        for tool_type, keywords in tool_keywords.items():
            if any(kw in step_lower for kw in keywords):
                ordered_tools.append(tool_type)
                break
    return ordered_tools


def _match_tool_to_type(tool_name: str) -> str:
    """Map actual OpenClaw tool names to our canonical types."""
    name = tool_name.lower().replace("_", "")
    mapping = {
        "websearch": "web_search", "search": "web_search",
        "webfetch": "web_fetch", "fetch": "web_fetch",
        "read": "read", "fileread": "read", "cat": "read",
        "write": "write", "filewrite": "write",
        "edit": "edit", "fileedit": "edit", "streplace": "edit",
        "exec": "exec", "bash": "exec", "shell": "exec",
    }
    return mapping.get(name, tool_name.lower())


def generic_rule_reward(
    turn_index: int,
    turn: dict,
    prev_turns: list[dict],
    all_turns: list[dict],
    task_id: str,
) -> float:
    """Mode B: rubric-guided rule-based process reward. Fast, no LLM call.

    Uses _get_task_rubric (TASK_RUBRICS + id aliases) for task-specific rewards:
    - Checks if tool usage follows reference_steps order (天眼)
    - Penalizes common_mistakes patterns from rubric
    - Rewards progress toward the task goal

    Score range per turn: [-0.5, +0.2]
    """
    content = turn.get("content", "")
    tool_name = _get_tool_name(turn)
    tool_args = _get_tool_args(turn)
    reward = 0.0

    rubric = _get_task_rubric(task_id)
    expected_tools = _parse_rubric_expected_tools(rubric)

    # Count which assistant turn this is (0-indexed)
    prev_assistant = [t for t in prev_turns if t.get("role") == "assistant"]
    assistant_seq = len(prev_assistant)

    if tool_name:
        reward += 0.10

        if tool_args and len(str(tool_args)) > 5:
            reward += 0.05

        # Check tool result from environment
        next_turns = all_turns[turn_index + 1:]
        for t in next_turns:
            if t.get("role") == "tool":
                tr_content = t.get("content", "")
                if tr_content.strip() and not _is_error_result(tr_content):
                    reward += 0.10
                elif _is_error_result(tr_content):
                    reward -= 0.05
                break
            elif t.get("role") == "assistant":
                break

        # Rubric-guided: check if tool matches expected step (天眼 alignment)
        cur_type = _match_tool_to_type(tool_name)
        if assistant_seq < len(expected_tools):
            expected = expected_tools[assistant_seq]
            if cur_type == expected:
                reward += 0.10  # following the reference path exactly
                logger.debug("Rubric match: turn %d tool=%s matches expected=%s",
                             assistant_seq, cur_type, expected)
            elif cur_type in expected_tools:
                reward += 0.03  # right tool but wrong order
        elif cur_type in expected_tools:
            reward += 0.05  # using a relevant tool type

        # Rubric-guided: check for common mistakes
        for mistake in rubric.get("common_mistakes", []):
            mistake_lower = mistake.lower()
            if "without reading" in mistake_lower or "without any web_search" in mistake_lower:
                # Penalize writing/acting before reading/searching
                if cur_type in ("write", "edit") and assistant_seq == 0:
                    if "read" in expected_tools[:2] or "web_search" in expected_tools[:2]:
                        reward -= 0.15
                        logger.debug("Rubric penalty: acting before gathering info")
            if "repeating" in mistake_lower and "same" in mistake_lower:
                pass  # handled below in repetition check
            if "fabricating" in mistake_lower or "without processing" in mistake_lower:
                # Penalize write on first turn without prior info gathering
                if cur_type == "write" and not any(
                    _match_tool_to_type(_get_tool_name(t) or "") in ("read", "web_search", "exec")
                    for t in prev_assistant if _get_tool_name(t)
                ):
                    reward -= 0.15
                    logger.debug("Rubric penalty: writing without info gathering")

    else:
        if not content.strip():
            reward -= 0.20
        elif assistant_seq == 0:
            # First turn with text-only (no tool call) is usually bad
            reward -= 0.10

    # Penalty for refusal / hallucination patterns
    hallucination_patterns = [r"I don'?t have access to", r"As an AI", r"I'?m unable to"]
    for p in hallucination_patterns:
        if re.search(p, content, re.IGNORECASE):
            reward -= 0.20
            break

    # Penalty for repeating same action as previous turn
    if prev_assistant and tool_name:
        prev_tool = _get_tool_name(prev_assistant[-1])
        prev_args = str(_get_tool_args(prev_assistant[-1]))
        cur_args = str(tool_args)
        if prev_tool == tool_name and prev_args == cur_args:
            reward -= 0.15

    return max(-0.5, min(0.2, reward))


# ══════════════════════════════════════════════════════════════
#  Main reward computation
# ══════════════════════════════════════════════════════════════

async def compute_episode_rewards_async(
    trajectory: list[dict[str, Any]],
    terminal_success: bool,
    task_id: str,
    task_prompt: str = "",
    mode: str = "self-judge",
    vllm_base_url: str = "http://localhost:9090/v1",
    judge_model: str = "Qwen3-4B",
    judge_api_key: str = "dummy",
) -> list[float]:
    """Compute per-assistant-turn rewards for a full episode (async version).

    Args:
        trajectory: list of message dicts (all roles)
        terminal_success: whether PinchBench grading passed
        task_id: PinchBench task ID
        task_prompt: original task prompt text
        mode: "baseline" (A), "rule" (B), "self-judge" (C), or "oracle-judge" (D)
        vllm_base_url: vLLM endpoint for self-judge
        judge_model: model name for judge calls
        judge_api_key: API key for judge calls

    Returns:
        List of rewards, one per assistant turn. Terminal reward added to last.
    """
    terminal_reward_raw = 1.0 if terminal_success else 0.0  # {0,+1}: no negative gradient for failures
    terminal_reward = TERMINAL_REWARD_WEIGHT * terminal_reward_raw

    assistant_indices = [
        i for i, t in enumerate(trajectory) if t.get("role") == "assistant"
    ]

    if not assistant_indices:
        return [terminal_reward]

    # Get expected number of turns from rubric
    rubric = _get_task_rubric(task_id)
    expected_turns = len(rubric.get("reference_steps", [4]))
    judge_model = await _resolve_judge_model_async(vllm_base_url, judge_model)

    rewards = []

    for seq_idx, turn_idx in enumerate(assistant_indices):
        turn = trajectory[turn_idx]
        prev_turns = trajectory[:turn_idx]

        if mode == "baseline":
            r = 0.0

        elif mode == "rule":
            r = generic_rule_reward(turn_idx, turn, prev_turns, trajectory, task_id)

        elif mode in ("self-judge", "oracle-judge"):
            tool_result = None
            for t in trajectory[turn_idx + 1:]:
                if t.get("role") == "tool":
                    tool_result = t.get("content", "")
                    break
                elif t.get("role") == "assistant":
                    break

            prm_prompt = build_prm_prompt(
                task_id=task_id,
                task_prompt=task_prompt,
                turn_index=seq_idx,
                total_turns=max(expected_turns, len(assistant_indices)),
                current_turn=turn,
                prev_turns=prev_turns,
                tool_result=tool_result,
                mode=mode,
            )

            base_url = vllm_base_url
            model = judge_model

            print(f"[PRM] Calling judge for turn {seq_idx}, url={base_url}, model={model}")
            r = await call_llm_judge(
                prm_prompt,
                vllm_base_url=base_url,
                model=model,
                api_key=judge_api_key,
            )
            print(f"[PRM] Judge returned: {r}")

        else:
            r = 0.0

        r = max(-0.5, min(0.2, r))
        rewards.append(r)

    # Add terminal reward to last turn
    rewards[-1] += terminal_reward

    return rewards


def compute_episode_rewards(
    trajectory: list[dict[str, Any]],
    terminal_success: bool,
    task_id: str,
    mode: str = "self-judge",
    task_prompt: str = "",
    vllm_base_url: str = "http://localhost:9090/v1",
    judge_model: str = "Qwen3-4B",
    judge_api_key: str = "dummy",
) -> list[float]:
    """Synchronous version of compute_episode_rewards.

    For modes requiring LLM calls, uses sync HTTP.
    """
    terminal_reward_raw = 1.0 if terminal_success else 0.0  # {0,+1}: no negative gradient for failures
    terminal_reward = TERMINAL_REWARD_WEIGHT * terminal_reward_raw

    assistant_indices = [
        i for i, t in enumerate(trajectory) if t.get("role") == "assistant"
    ]

    if not assistant_indices:
        return [terminal_reward]

    rubric = _get_task_rubric(task_id)
    expected_turns = len(rubric.get("reference_steps", [4]))
    judge_model = _resolve_judge_model_sync(vllm_base_url, judge_model)

    rewards = []

    for seq_idx, turn_idx in enumerate(assistant_indices):
        turn = trajectory[turn_idx]
        prev_turns = trajectory[:turn_idx]

        if mode == "baseline":
            r = 0.0

        elif mode == "rule":
            r = generic_rule_reward(turn_idx, turn, prev_turns, trajectory, task_id)

        elif mode in ("self-judge", "oracle-judge"):
            tool_result = None
            for t in trajectory[turn_idx + 1:]:
                if t.get("role") == "tool":
                    tool_result = t.get("content", "")
                    break
                elif t.get("role") == "assistant":
                    break

            prm_prompt = build_prm_prompt(
                task_id=task_id,
                task_prompt=task_prompt,
                turn_index=seq_idx,
                total_turns=max(expected_turns, len(assistant_indices)),
                current_turn=turn,
                prev_turns=prev_turns,
                tool_result=tool_result,
                mode=mode,
            )

            r = call_llm_judge_sync(
                prm_prompt,
                vllm_base_url=vllm_base_url,
                model=judge_model,
                api_key=judge_api_key,
            )

        else:
            r = 0.0

        r = max(-0.5, min(0.2, r))
        rewards.append(r)

    rewards[-1] += terminal_reward
    return rewards


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> float:
    """veRL-compatible reward function entry point (terminal only)."""
    if extra_info is None:
        extra_info = {}
    terminal_success = bool(extra_info.get("terminal_success", False))
    return 1.0 if terminal_success else 0.0  # {0,+1}: no negative gradient for failures
