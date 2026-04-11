"""
Process reward + terminal reward computation for PinchBench RL.

Four ablation modes:
  Mode A (baseline):    per-turn reward = 0, last turn gets terminal_reward
  Mode B (rule-only):   per-turn reward from generic behavior rules (fast, no LLM)
  Mode C (self-judge):  Qwen3-4B self-judge with goal + optional hints + common mistakes
  Mode D (oracle-judge): qwen-plus judge (fallback if self-judge is unreliable)

Default mode: C (self-judge) — 自进化：模型自己评判自己

Terminal reward: {-1, +1}  (task fail / succeed)
Process reward:  [-0.5, +0.3] per turn

PRM (self-judge) uses task goal + optional_hints (non-binding) + common_mistakes.
reference_steps in TASK_RUBRICS is for Mode B rule rewards only, not verbatim PRM text.
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
    "Typical agents gather context, take actions, and verify; many valid tool orders and strategies "
    "can satisfy the task goal."
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
            "Often: search the web for AAPL, optionally fetch a page for detail, then write stock_report.txt "
            "with ticker, price, date, and summary. Order can vary if the file is grounded and meets graders."
        ),
        "reference_steps": [
            "1. web_search: search for AAPL/Apple stock price",
            "2. web_fetch or additional web_search: get detailed data (may retry different sources)",
            "3. write: create stock_report.txt with price ($xxx.xx), date, market summary, and the ticker symbol AAPL as text",
            "4. Summarize and confirm completion",
        ],
        "common_mistakes": [
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
            "Typically: list/read config files, apply every required substitution in both JSON and YAML, "
            "then verify. Different edit tools or read order are fine if final files match the task."
        ),
        "reference_steps": [
            "1. read or exec: list config/ and read both files",
            "2. Use OpenClaw edit tool (or careful search_replace) — avoid blind sed without reading",
            "3. Apply all replacement rules across JSON and YAML",
            "4. read: verify replacements; summarize changes",
        ],
        "common_mistakes": [
            "Running sed -i without reading targets first (blind replacement)",
            "Repeating the same failing exec/sed instead of switching to edit",
            "Missing one file or one replacement class (localhost, DB name, log level, API URL)",
            "Not verifying with read after edits",
        ],
        "qwen_plus_stats": "9 turns, 8 tool calls. Used edit with expanded context for ambiguous matches.",
    },
    "task_10_workflow": {
        "goal": "Read workspace config.json, extract API endpoint, write a Python script that calls it, document in NOTES.md (same as PinchBench task; hybrid grader checks script + notes quality).",
        "optional_hints": (
            "Usually: read config.json before coding, then produce script + NOTES. Iteration is normal; "
            "the grader penalizes hardcoding the URL without using the config."
        ),
        "reference_steps": [
            "1. read: read config.json to extract endpoint and settings",
            "2. Plan: identify endpoint URL, method, headers from JSON",
            "3. write: Python script with requests (or urllib), error handling, uses endpoint from config",
            "4. write: NOTES.md explaining setup and how to run the script",
        ],
        "common_mistakes": [
            "Hardcoding the URL instead of reading config.json (still common; judge penalizes)",
            "Script without basic error handling",
            "Missing NOTES.md or notes that omit what was done",
        ],
        "qwen_plus_stats": "4 turns, 3 tool calls. Python: 1704 bytes, NOTES: 1389 bytes.",
    },
    "task_22_second_brain": {
        "goal": "Multi-session task (OpenClaw sessions in frontmatter): Session 1 — write user facts to memory/MEMORY.md (Rust, Jan 15 2024, Dr. Elena Vasquez, NeonDB, secret phrase). Session 2 — answer language + project from file. Session 3 (new session) — read MEMORY.md and answer all 5 recall questions. Matches PinchBench hybrid grader.",
        "optional_hints": (
            "Three OpenClaw sessions: persist facts to memory/MEMORY.md, answer short questions, then in a "
            "new session read the file and answer five recall items. Wording inside a session may vary."
        ),
        "reference_steps": [
            "1. Session 1: write memory/MEMORY.md with required facts; confirm save",
            "2. Session 2: answer Rust + NeonDB (read file if needed)",
            "3. Session 3: read MEMORY.md; answer 5 questions accurately (language, date, mentor, project description, code phrase)",
        ],
        "common_mistakes": [
            "Wrong path (must be memory/MEMORY.md under workspace)",
            "Session 3 not reading file and hallucinating answers",
            "Omitting one of the five recall items",
        ],
        "qwen_plus_stats": "Multi-session; ~3 OpenClaw session invocations per benchmark run.",
    },
    "task_16_email_triage": {
        "goal": "Read all 13 files in inbox/ (email_01.txt … email_13.txt), then write triage_report.md: top summary + day plan; for each email assign Priority P0–P4, Category (incident/client/internal-request/administrative/code-review/automated/newsletter/spam), and recommended action; sort entries by priority (most urgent first). PinchBench checks: production outage email as P0; monitoring alert tied to same incident; BigClient email P0/P1; promotional/spam email P4; report structure and summary section.",
        "optional_hints": (
            "Commonly: read all inbox emails (batch or sequential), then write triage_report.md with a top "
            "summary and priority-sorted rows. Reading order is not graded; coverage and priorities are."
        ),
        "reference_steps": [
            "1. read/exec: discover inbox/ and enumerate email_01 … email_13",
            "2. read: open each email file (one or more read calls per email until all 13 understood)",
            "3. Plan: assign P0–P4 and category per grading rules (incident vs client vs spam, etc.)",
            "4. write: triage_report.md with summary block first, then sections sorted by priority, each row with priority/category/action",
        ],
        "common_mistakes": [
            "Writing triage_report.md before reading all 13 emails or omitting any email from the report",
            "Missing file triage_report.md or wrong filename",
            "Production database outage / CTO war-room email not labeled P0",
            "API latency monitoring alert (email 13) not linked or grouped with the DB outage incident",
            "BigClient / $2M contract email not P0 or P1",
            "Promotional flash-sale / spam-like email not P4 or lowest tier",
            "Report not sorted by priority (most urgent first) or no executive summary + day plan at the top",
            "Missing per-email recommended action or category keywords the grader expects",
        ],
        "qwen_plus_stats": "15 turns, 14 tool calls; read each inbox file; triage_report.md with summary then priority-sorted entries.",
    },
    "task_18_spreadsheet_summary": {
        "goal": "Analyze workspace CSV + XLSX per task, compute real aggregates, write data_summary.md (PinchBench id=task_18_spreadsheet_summary; markdown file may be task_19_spreadsheet_summary.md on disk).",
        "optional_hints": (
            "Usually: inspect CSV, run shell/python to extract XLSX numbers, compute aggregates, then write "
            "data_summary.md. pandas vs awk vs other tools is fine if numbers match command output."
        ),
        "reference_steps": [
            "1. read: CSV text; XLSX may be binary — use exec (python/awk) not raw paste",
            "2. exec: compute sums/means/top-N with verifiable commands",
            "3. write: data_summary.md with numbers that match exec output",
        ],
        "common_mistakes": [
            "Pretending to read .xlsx as UTF-8 text and inventing numbers",
            "Writing summary without any successful numeric extraction",
            "Not retrying with another tool when first exec fails",
        ],
        "qwen_plus_stats": "7 turns, 6 tool calls, 1728 bytes. Switched from pandas to awk when needed.",
    },
    "task_18_market_research": {
        "goal": "Write market_research.md: enterprise observability/APM landscape, top ~5 players, differentiators, pricing models, trends; comparison table and analyst-style structure. Use web search if available; else knowledge (per task text).",
        "optional_hints": (
            "Often: one or more web searches, then a structured market_research.md. Depth and number of "
            "searches vary; coverage and analyst-style sections matter more than a fixed recipe."
        ),
        "reference_steps": [
            "1. web_search (if skills available): market overview, competitors, pricing",
            "2. Optional additional searches for specificity",
            "3. write: market_research.md with executive summary, profiles, table, trends",
        ],
        "common_mistakes": [
            "Missing market_research.md or fewer than 5 meaningful competitor sections",
            "Generic fluff with no pricing or trends",
            "Search queries with stale years when using web_search",
        ],
        "qwen_plus_stats": "5 turns, 4 tool calls, 8412 bytes. Three search angles: leaders, comparison, pricing.",
    },
    "task_24_polymarket_briefing": {
        "goal": "Write polymarket_briefing.md: top 3 trending Polymarket markets (real/active), Yes/No odds, related news within 48h. Task allows gamma API or polymarket.com / web search; do not fabricate markets or odds.",
        "optional_hints": (
            "Typically: discover three real trending markets (API or web), pair each with recent news, then "
            "write the briefing. Do not invent market titles or odds; sources may differ."
        ),
        "reference_steps": [
            "1. web_search or fetch: trending Polymarket markets (or API per task description)",
            "2. For each of 3 markets: find corroborating recent news",
            "3. write: polymarket_briefing.md with dated header and 3 sections per template",
        ],
        "common_mistakes": [
            "Hallucinated market names or odds",
            "News not tied to the market or not recent",
            "Wrong filename or missing 3 market blocks",
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

    prompt = f"""You are an AI Agent behavior evaluator. Score the current step of an agent working on a task.

## Task Goal
{rubric.get("goal", task_prompt[:500])}

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
Evaluate whether this step moves the agent closer to the goal.
Consider:
- Is this step useful and appropriate given prior context? (Do **not** treat optional hints as a mandatory script; reward valid alternative strategies.)
- Is the agent avoiding the common mistakes listed above?
- Is this step coherent with what the agent has already done?

Score range: -0.5 (harmful/wasteful step) to +0.3 (excellent progress)
- +0.2 to +0.3: Step clearly advances toward the goal (e.g., correct tool with good arguments)
- +0.05 to +0.15: Reasonable step, some progress
- 0.0: Neutral, neither helpful nor harmful
- -0.1 to -0.2: Suboptimal but not terrible (e.g., redundant action)
- -0.3 to -0.5: Actively harmful (e.g., fabricating data, repeating failed commands)

Respond with ONLY a JSON object, no other text:
{"score": <float between -0.5 and 0.3>, "reason": "<one sentence explanation>"}"""

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

    Returns score in [-0.5, +0.3]. Falls back to 0.0 on any error.
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
        return max(-0.5, min(0.3, score))
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
        return max(-0.5, min(0.3, score))
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

    Score range per turn: [-0.5, +0.3]
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

    return max(-0.5, min(0.3, reward))


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
    terminal_reward_raw = 1.0 if terminal_success else -1.0
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

            rule_r = generic_rule_reward(turn_idx, turn, prev_turns, trajectory, task_id)
            if r == 0.0:
                r = rule_r
                print(f"[PRM] Judge returned 0.0, fallback to rule: {r:.2f}")
            else:
                blended = 0.7 * r + 0.3 * rule_r
                print(f"[PRM] Judge={r:.2f}, rule={rule_r:.2f}, blended={blended:.2f}")
                r = blended

        else:
            r = 0.0

        r = max(-0.5, min(0.3, r))
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
    terminal_reward_raw = 1.0 if terminal_success else -1.0
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

        r = max(-0.5, min(0.3, r))
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
    return 1.0 if terminal_success else -1.0
