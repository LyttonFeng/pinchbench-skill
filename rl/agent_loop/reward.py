"""
Process reward + terminal reward computation for PinchBench RL.

Four ablation modes:
  Mode A (baseline):    per-turn reward = 0, last turn gets terminal_reward
  Mode B (rule-only):   per-turn reward from generic behavior rules (fast, no LLM)
  Mode C (self-judge):  Qwen3-4B self-judge with rubric + reference trajectory
  Mode D (oracle-judge): qwen-plus judge (fallback if self-judge is unreliable)

Default mode: C (self-judge) — 自进化：模型自己评判自己

Terminal reward: {-1, +1}  (task fail / succeed)
Process reward:  [-0.5, +0.3] per turn
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    logger.addHandler(_h)


# ══════════════════════════════════════════════════════════════
#  Per-task PRM rubrics (reference trajectories + scoring guide)
# ══════════════════════════════════════════════════════════════

TASK_RUBRICS: dict[str, dict[str, Any]] = {
    "task_02_stock": {
        "goal": "Research Apple (AAPL) stock price, save to stock_report.txt with price, date, and market summary.",
        "reference_steps": [
            "1. web_search: search for AAPL/Apple stock price",
            "2. web_fetch or additional web_search: get detailed data (may retry different sources)",
            "3. write: create stock_report.txt with price ($xxx.xx), date, and market summary",
            "4. Summarize and confirm completion",
        ],
        "common_mistakes": [
            "Searching but never writing the file (premature termination)",
            "Writing the file without any web_search first (fabricating data)",
            "Repeating the same failed search query without changing it",
        ],
        "qwen_plus_stats": "7 turns, 6 tool calls, 745 bytes output",
    },
    "task_12_skill_search": {
        "goal": "Find and replace specific values in config files (settings.json and database.yml).",
        "reference_steps": [
            "1. exec ls or read: examine directory structure and file contents",
            "2. read: read each config file to understand current values",
            "3. edit: precisely replace target values (NOT blind sed)",
            "4. read: verify the modifications were applied correctly",
        ],
        "common_mistakes": [
            "Using sed -i without reading the file first (blind replacement)",
            "Repeating the same failing sed command multiple times",
            "Not verifying changes after editing",
            "Using sed on macOS without -i '' (syntax error)",
        ],
        "qwen_plus_stats": "9 turns, 8 tool calls. Used edit with expanded context for ambiguous matches.",
    },
    "task_10_workflow": {
        "goal": "Read config.json, extract API endpoint, create a Python script to call it, document in NOTES.md.",
        "reference_steps": [
            "1. read: read config.json to extract endpoint and settings",
            "2. Analyze: identify endpoint (api.example.com), method, headers",
            "3. write: create Python script with requests, json, error handling",
            "4. write: create NOTES.md documenting the workflow",
        ],
        "common_mistakes": [
            "Writing the Python script without reading config.json first",
            "Python script missing error handling (no try/except)",
            "NOTES.md too brief (less than 500 bytes)",
        ],
        "qwen_plus_stats": "4 turns, 3 tool calls. Python: 1704 bytes, NOTES: 1389 bytes.",
    },
    "task_22_second_brain": {
        "goal": "Create and organize knowledge notes with cross-references and structured format.",
        "reference_steps": [
            "1. read/exec: examine existing note structure",
            "2. read: read existing note content to understand context",
            "3. write: create new notes with structured format (headers, lists, links)",
            "4. write: update index or create cross-references between notes",
        ],
        "common_mistakes": [
            "Writing notes without reading existing content first",
            "Notes lacking structure (no headers, lists, or links)",
            "Only creating one file without cross-references",
        ],
        "qwen_plus_stats": "7 turns, 4 tool calls.",
    },
    "task_16_email_triage": {
        "goal": "Read all emails, analyze priority, and write classification/triage results.",
        "reference_steps": [
            "1. read/exec: list email directory contents",
            "2. read: read each email individually (multiple read calls)",
            "3. Analyze: discuss priority and categorization",
            "4. write: write classification results covering all emails",
        ],
        "common_mistakes": [
            "Reading only some emails before writing conclusions",
            "Not covering all emails in the triage output",
            "Skipping the analysis step and going straight to writing",
        ],
        "qwen_plus_stats": "15 turns, 14 tool calls. Read every email individually.",
    },
    "task_19_spreadsheet_summary": {
        "goal": "Analyze CSV/XLSX data and write a summary report with actual computed values.",
        "reference_steps": [
            "1. read: read CSV file to understand data structure",
            "2. read: attempt to read XLSX (will get binary)",
            "3. exec: use awk/python/pandas to compute actual statistics",
            "4. write: write report referencing real computed values",
        ],
        "common_mistakes": [
            "Reading XLSX binary and writing report without processing the data",
            "Report contains fabricated numbers not matching exec results",
            "Not switching strategy when pandas is unavailable (try awk instead)",
        ],
        "qwen_plus_stats": "7 turns, 6 tool calls, 1728 bytes. Switched from pandas to awk when needed.",
    },
    "task_18_market_research": {
        "goal": "Research APM/observability market: overview, competitors, pricing. Write comprehensive report.",
        "reference_steps": [
            "1. web_search: search market overview (leaders, market size)",
            "2. web_search: search competitor comparison (Datadog vs Splunk etc.)",
            "3. web_search: search pricing models",
            "4. write: write comprehensive report (>5000 bytes) covering all dimensions",
        ],
        "common_mistakes": [
            "Only searching once before writing the report",
            "Using outdated year in search queries (2023 or earlier)",
            "Report too short (less than 2000 bytes)",
            "Missing competitor comparison or pricing analysis",
        ],
        "qwen_plus_stats": "5 turns, 4 tool calls, 8412 bytes. Three search angles: leaders, comparison, pricing.",
    },
    "task_24_polymarket_briefing": {
        "goal": "Research Polymarket trending markets, search news for each, write briefing with 3 markets.",
        "reference_steps": [
            "1. web_search: search Polymarket trending/popular markets",
            "2. web_search: search news for specific market topic 1",
            "3. web_search: search news for specific market topic 2-3",
            "4. write: write briefing covering 3 markets with sources",
        ],
        "common_mistakes": [
            "Using outdated year in search queries (2023 or earlier)",
            "Fabricating market probabilities without search evidence",
            "Not searching for individual market topics (only general search)",
            "Report missing specific sections for each market (## 1, ## 2, ## 3)",
        ],
        "qwen_plus_stats": "8 turns, 7 tool calls, 1719 bytes. Searched trends then each topic individually.",
    },
}

# Generic rubric for unknown tasks
_GENERIC_RUBRIC = {
    "goal": "Complete the assigned task using appropriate tools.",
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
      - Task goal and rubric (what success looks like)
      - Reference trajectory (天眼: how qwen-plus succeeded)
      - Common mistakes to watch for
      - The agent's previous actions (context)
      - The current turn to evaluate
    """
    rubric = TASK_RUBRICS.get(task_id, _GENERIC_RUBRIC)

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

    # Build reference steps string
    ref_steps = "\n".join(rubric.get("reference_steps", []))

    # Build common mistakes string
    mistakes = "\n".join(f"- {m}" for m in rubric.get("common_mistakes", []))

    # Format tool args for display
    args_display = json.dumps(tool_args, ensure_ascii=False, indent=2)[:300] if tool_args else "(none)"

    prompt = f"""You are an AI Agent behavior evaluator. Score the current step of an agent working on a task.

## Task Goal
{rubric.get("goal", task_prompt[:500])}

## Reference Path (how a successful agent solves this)
{ref_steps}

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
- Is the agent following the reference path or doing something useful?
- Is the agent avoiding the common mistakes listed above?
- Is this step appropriate given what the agent has already done?

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

    Returns score in [-0.5, +0.3]. Falls back to 0.0 on any error.
    """
    import aiohttp

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict scoring function. Respond with ONLY a JSON object."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 128,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{vllm_base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    logger.warning("PRM judge HTTP %d", resp.status)
                    return 0.0
                data = await resp.json()
    except Exception as e:
        logger.warning("PRM judge request failed: %s", e)
        return 0.0

    try:
        text = data["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # Strip <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        result = json.loads(text)
        score = float(result.get("score", 0.0))
        reason = result.get("reason", "")
        logger.debug("PRM judge: score=%.2f reason=%s", score, reason)
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
    """Synchronous wrapper for call_llm_judge."""
    from urllib import request as urllib_request, error as urllib_error

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict scoring function. Respond with ONLY a JSON object."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 128,
    }).encode("utf-8")

    req = urllib_request.Request(
        f"{vllm_base_url}/chat/completions",
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
        text = data["choices"][0]["message"]["content"].strip()
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

    Uses TASK_RUBRICS to give task-specific rewards:
    - Checks if tool usage follows reference_steps order (天眼)
    - Penalizes common_mistakes patterns from rubric
    - Rewards progress toward the task goal

    Score range per turn: [-0.5, +0.3]
    """
    content = turn.get("content", "")
    tool_name = _get_tool_name(turn)
    tool_args = _get_tool_args(turn)
    reward = 0.0

    rubric = TASK_RUBRICS.get(task_id, _GENERIC_RUBRIC)
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
    terminal_reward = 1.0 if terminal_success else -1.0

    assistant_indices = [
        i for i, t in enumerate(trajectory) if t.get("role") == "assistant"
    ]

    if not assistant_indices:
        return [terminal_reward]

    # Get expected number of turns from rubric
    rubric = TASK_RUBRICS.get(task_id, _GENERIC_RUBRIC)
    expected_turns = len(rubric.get("reference_steps", [4]))

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
    terminal_reward = 1.0 if terminal_success else -1.0

    assistant_indices = [
        i for i, t in enumerate(trajectory) if t.get("role") == "assistant"
    ]

    if not assistant_indices:
        return [terminal_reward]

    rubric = TASK_RUBRICS.get(task_id, _GENERIC_RUBRIC)
    expected_turns = len(rubric.get("reference_steps", [4]))

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
