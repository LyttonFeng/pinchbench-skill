#!/usr/bin/env python3
"""Harness-driven focused DPO data generation for task_18_spreadsheet_summary.

Real topology:
  Mac/Control host
    - runs ModelProxy
    - forwards proxy requests to vLLM
  ECS
    - runs OpenClaw through SSH reverse tunnel
  L40S / remote inference host
    - runs vLLM with tool parser

This collector captures *real* first-step failures from the runtime harness and
converts them into focused DPO pairs:
  chosen   = rule-based exec+pandas first action
  rejected = real first assistant action from the failed rollout
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import aiohttp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_tasks import Task, TaskLoader  # noqa: E402
from rl.agent_loop.model_proxy import ModelProxy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_copy_with_prompt(task: Task, prompt: str) -> Task:
    return Task(
        task_id=task.task_id,
        name=task.name,
        category=task.category,
        grading_type=task.grading_type,
        timeout_seconds=task.timeout_seconds,
        workspace_files=copy.deepcopy(task.workspace_files),
        prompt=prompt,
        expected_behavior=task.expected_behavior,
        grading_criteria=list(task.grading_criteria),
        automated_checks=task.automated_checks,
        llm_judge_rubric=task.llm_judge_rubric,
        grading_weights=copy.deepcopy(task.grading_weights),
        file_path=task.file_path,
        frontmatter=copy.deepcopy(task.frontmatter),
    )


def _make_prompt_variants(base_prompt: str, target_count: int) -> list[str]:
    intros = [
        "Please analyze the files below and write the required report.",
        "I need a clean analysis of the two data files in the workspace.",
        "Analyze these workspace data files and produce the requested report.",
        "Please read both files carefully and write a summary report.",
        "Use the provided files to generate the requested markdown summary.",
    ]
    nudges = [
        "Do not guess spreadsheet contents without reading the workbook structure.",
        "Make sure the Excel workbook is actually parsed before summarizing it.",
        "The workbook has multiple sheets, so inspect it structurally before reporting numbers.",
        "Be careful not to treat the Excel workbook as plain text.",
        "Use the available tools to inspect the real file contents before writing conclusions.",
    ]
    closers = [
        "Keep the report concise but correct.",
        "Accuracy matters more than speed.",
        "Do the real computation before writing `data_summary.md`.",
        "Use the workspace files directly; do not make assumptions.",
        "Write the final report only after the required analysis is complete.",
    ]

    prompts: list[str] = []
    for i in range(target_count):
        intro = intros[i % len(intros)]
        nudge = nudges[(i // len(intros)) % len(nudges)]
        closer = closers[(i // (len(intros) * len(nudges))) % len(closers)]
        prompts.append(f"{intro}\n\n{base_prompt}\n\nNote: {nudge}\n{closer}")
    return prompts[:target_count]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "thinking":
                parts.append(str(item.get("thinking", "")))
    return "\n".join(parts)


def _strip_workspace_path(text: str, workspace: str) -> str:
    if not workspace:
        return text
    for path in [workspace, str(Path(workspace).parent)]:
        p = path.rstrip("/")
        text = text.replace(p + "/", "")
        text = text.replace(p, ".")
    return text


def _transcript_to_messages(transcript: list[dict[str, Any]], workspace: str = "") -> list[dict[str, Any]]:
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


def _first_assistant_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg
    return None


def _first_negative_type(msg: dict[str, Any] | None) -> str | None:
    if not msg:
        return "no_assistant"
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        content = str(msg.get("content", "")).lower()
        if "who am i" in content or "just came online" in content:
            return "bootstrap_chatter"
        return "no_tool_call"
    fn = tool_calls[0].get("function", {})
    name = str(fn.get("name", "")).lower()
    raw_args = str(fn.get("arguments", "")).lower()
    if name == "read" and "company_expenses.xlsx" in raw_args:
        return "read_xlsx"
    if name == "session_status":
        return "session_status"
    if name == "read" and "quarterly_sales.csv" in raw_args:
        return "read_csv_first"
    if name == "exec" and any(x in raw_args for x in ["ls -la", "pwd && ls", "find . -maxdepth"]):
        return "explore_ls"
    return None


def _chosen_exec_message(variant: int) -> dict[str, Any]:
    commands = [
        (
            "python - <<'PY'\n"
            "import pandas as pd\n"
            "xls = pd.ExcelFile('company_expenses.xlsx')\n"
            "print({'sheets': xls.sheet_names})\n"
            "for name, df in pd.read_excel('company_expenses.xlsx', sheet_name=None).items():\n"
            "    print(f'--- {name} ---')\n"
            "    print(df.head())\n"
            "PY"
        ),
        (
            "python - <<'PY'\n"
            "import pandas as pd\n"
            "book = pd.read_excel('company_expenses.xlsx', sheet_name=None)\n"
            "print({'sheets': list(book)})\n"
            "for name, df in book.items():\n"
            "    print(f'--- {name} ---')\n"
            "    print(df.columns.tolist())\n"
            "    print(df.head())\n"
            "PY"
        ),
        (
            "python - <<'PY'\n"
            "import pandas as pd\n"
            "xls = pd.ExcelFile('company_expenses.xlsx')\n"
            "print({'sheets': xls.sheet_names})\n"
            "for sheet in xls.sheet_names:\n"
            "    df = pd.read_excel('company_expenses.xlsx', sheet_name=sheet)\n"
            "    print(f'--- {sheet} ---')\n"
            "    print({'shape': df.shape, 'columns': df.columns.tolist()})\n"
            "    print(df.head())\n"
            "PY"
        ),
    ]
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_exec_{variant}",
                "type": "function",
                "function": {
                    "name": "exec",
                    "arguments": json.dumps({"command": commands[variant % len(commands)]}, ensure_ascii=False),
                },
            }
        ],
    }


def _slugify_model(model_id: str) -> str:
    return model_id.replace("/", "-").replace(".", "-").lower()


def _remote_parts() -> tuple[str, str, str, str]:
    host = (os.environ.get("OPENCLAW_HOST") or "").strip()
    if not host:
        raise SystemExit("OPENCLAW_HOST is required for harness-driven collector")
    user = os.environ.get("OPENCLAW_USER", "root")
    port = os.environ.get("OPENCLAW_PORT", "22")
    ssh_key = os.environ.get("OPENCLAW_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))
    return host, user, port, ssh_key


def _remote_activate_prefix() -> str:
    return os.environ.get("OPENCLAW_REMOTE_ACTIVATE_CMD", "").strip()


def _rsync_bin() -> str:
    return shutil.which("rsync") or "rsync"


def _prepare_local_seed(task: Task) -> Path:
    td = tempfile.mkdtemp(prefix="task18_seed_")
    root = Path(td)
    for file_spec in task.workspace_files:
        if "content" in file_spec:
            dest = root / file_spec["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(file_spec["content"], encoding="utf-8")
        elif "source" in file_spec:
            src = REPO_ROOT / "assets" / file_spec["source"]
            dest = root / file_spec.get("dest", file_spec["source"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
    return root


def _seed_remote_workspace(task: Task, remote_workspace: str) -> None:
    host, user, port, ssh_key = _remote_parts()
    ws = shlex.quote(remote_workspace)
    reset = subprocess.run(
        [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            "-i", ssh_key, "-p", str(port), f"{user}@{host}",
            f"rm -rf {ws} && mkdir -p {ws}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if reset.returncode != 0:
        raise RuntimeError(f"remote workspace reset failed: {reset.stderr.strip() or reset.stdout.strip()}")
    seed_root = _prepare_local_seed(task)
    try:
        sync = subprocess.run(
            [
                _rsync_bin(),
                "-az",
                "--timeout=60",
                "-e",
                f"ssh -o StrictHostKeyChecking=no -i {ssh_key} -p {port}",
                f"{seed_root}/",
                f"{user}@{host}:{remote_workspace}/",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if sync.returncode != 0:
            raise RuntimeError(f"remote workspace rsync failed: {sync.stderr.strip() or sync.stdout.strip()}")
    finally:
        shutil.rmtree(seed_root, ignore_errors=True)


def _build_remote_setup(agent_id: str, proxy_url: str, remote_workspace: str) -> str:
    models_json = json.dumps({
        "mode": "replace",
        "providers": {"verl": {
            "baseUrl": proxy_url,
            "apiKey": "dummy",
            "api": "openai-completions",
            "models": [{"id": "verl-proxy", "name": "verl-proxy", "reasoning": False}],
        }},
        "defaultProvider": "verl",
        "defaultModel": "verl/verl-proxy",
    }, indent=2)
    auth_json = json.dumps({
        "version": 1,
        "profiles": {"verl-default": {"type": "api_key", "key": "dummy", "provider": "verl"}},
    }, indent=2)
    b64_models = base64.b64encode(models_json.encode()).decode()
    b64_auth = base64.b64encode(auth_json.encode()).decode()
    get_agent_dir_py = (
        f'python3 -c "'
        f'import json; '
        f'd=json.load(open(\\\"$HOME/.openclaw/openclaw.json\\\")); '
        f'ms=[a for a in d[\\\"agents\\\"][\\\"list\\\"] if a.get(\\\"name\\\")==\\\"{agent_id}\\\" or a.get(\\\"id\\\")==\\\"{agent_id}\\\"]; '
        f'print(ms[0][\\\"agentDir\\\"]) if ms else print(\\\"$HOME/.openclaw/agents/{agent_id}/agent\\\")'
        f'"'
    )
    return " && ".join([
        f"mkdir -p {shlex.quote(remote_workspace)}",
        f"openclaw agents add {agent_id} --model verl/verl-proxy --workspace {shlex.quote(remote_workspace)} --non-interactive 2>/dev/null || true",
        f"_adir=$({get_agent_dir_py})",
        "mkdir -p $_adir",
        f"echo {b64_models} | base64 -d > $_adir/models.json",
        f"echo {b64_auth} | base64 -d > $_adir/auth-profiles.json",
        "rm -f $_adir/../sessions/sessions.json",
    ])


async def _forward_to_vllm(
    vllm_base_url: str,
    model: str,
    messages: list,
    tools: list | None,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": min(max_tokens, 4096),
        "temperature": 0.0,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    endpoint = vllm_base_url.rstrip("/") + "/chat/completions"
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            return await resp.json()


async def _fetch_remote_transcript(agent_id: str, session_id: str) -> list[dict[str, Any]]:
    host, user, port, ssh_key = _remote_parts()
    proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "StrictHostKeyChecking=no", "-i", ssh_key, "-p", str(port),
        f"{user}@{host}",
        f"cat $HOME/.openclaw/agents/{agent_id}/sessions/{session_id}.jsonl",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
    transcript: list[dict[str, Any]] = []
    for line in out.decode(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            transcript.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return transcript


async def _run_remote_task_via_proxy(
    *,
    task: Task,
    agent_id: str,
    run_id: str,
    model: str,
    vllm_base_url: str,
    verbose: bool = False,
) -> dict[str, Any]:
    host, user, port, ssh_key = _remote_parts()
    proxy = ModelProxy(host="127.0.0.1", port=0, timeout=max(task.timeout_seconds, 300))
    proxy_port = await proxy.start()
    remote_workspace = f"/tmp/pinchbench/{run_id}/agent_workspace"
    session_id = f"{task.task_id}_{int(time.time() * 1000)}"
    _seed_remote_workspace(task, remote_workspace)

    local_proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
    setup_cmd = _build_remote_setup(agent_id, local_proxy_url, remote_workspace)
    escaped_prompt = task.prompt.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    run_cmd = f'openclaw agent --agent {agent_id} --session-id {session_id} --message "{escaped_prompt}" --local'
    remote_prefix = _remote_activate_prefix()
    remote_command = " && ".join(filter(None, [remote_prefix, setup_cmd, f"cd {shlex.quote(remote_workspace)} && {run_cmd}"]))

    proc = await asyncio.create_subprocess_exec(
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "ExitOnForwardFailure=yes",
        "-R", f"{proxy_port}:127.0.0.1:{proxy_port}",
        "-i", ssh_key,
        "-p", str(port),
        f"{user}@{host}",
        remote_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    first_request_messages: list[dict[str, Any]] | None = None
    first_request_tools: list[dict[str, Any]] | None = None
    turn_count = 0
    started_at = time.time()
    try:
        while turn_count < 20:
            try:
                req = await asyncio.wait_for(proxy.get_request(), timeout=180 if turn_count == 0 else 90)
            except asyncio.TimeoutError:
                break
            if first_request_messages is None:
                first_request_messages = copy.deepcopy(req.messages)
                first_request_tools = copy.deepcopy(req.tools or [])
            vllm_resp = await _forward_to_vllm(vllm_base_url, model, req.messages, req.tools, req.max_tokens)
            if "error" in vllm_resp:
                req.response_error = str(vllm_resp["error"])
                await proxy.send_response(req)
                break
            choice = vllm_resp["choices"][0]
            msg = choice["message"]
            req.response_text = msg.get("content", "")
            req.finish_reason = choice.get("finish_reason", "stop")
            req.response_usage = vllm_resp.get("usage")
            if msg.get("tool_calls"):
                req.response_tool_calls = msg["tool_calls"]
            await proxy.send_response(req)
            turn_count += 1
            if turn_count >= 1:
                # For focused DPO we only need the first assistant action.
                # Give OpenClaw a short window to persist the first-turn transcript,
                # then stop the remote run instead of waiting for the whole task.
                await asyncio.sleep(3)
                break
    finally:
        try:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.communicate(), timeout=10)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
            else:
                await asyncio.wait_for(proc.communicate(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
        transcript = await _fetch_remote_transcript(agent_id, session_id)
        await proxy.stop()

    return {
        "status": "success" if transcript else "error",
        "transcript": transcript,
        "workspace": remote_workspace,
        "execution_time": time.time() - started_at,
        "first_request_messages": first_request_messages or [],
        "first_request_tools": first_request_tools or [],
    }


async def collect_async(args: argparse.Namespace) -> None:
    tasks_dir = REPO_ROOT / "tasks"
    loader = TaskLoader(tasks_dir)
    task = loader.load_task(loader.tasks_dir / "task_19_spreadsheet_summary.md")
    prompt_variants = _make_prompt_variants(task.prompt, args.target_pairs * 3)

    output_path = Path(args.output)
    transcript_dir = Path(args.transcript_dir)
    rows: list[dict[str, Any]] = []
    negative_counts: Counter[str] = Counter()
    model_slug = _slugify_model(args.model)
    prompt_template = _load_json(REPO_ROOT / "rl/data/generated/task_18_spreadsheet_summary_runtime/runtime_prompt_template.json")
    agent_id = args.agent_id or f"task18-focused-dpo-{model_slug}"

    for idx, prompt in enumerate(prompt_variants):
        if len(rows) >= args.target_pairs:
            break
        run_id = f"{args.run_id_start + idx}-focused-{model_slug}"
        task_variant = _task_copy_with_prompt(task, prompt)
        result = await _run_remote_task_via_proxy(
            task=task_variant,
            agent_id=agent_id,
            run_id=run_id,
            model=args.model,
            vllm_base_url=args.vllm_base_url,
            verbose=args.verbose,
        )
        transcript = result.get("transcript", [])
        transcript_path = transcript_dir / model_slug / f"variant_{idx:04d}.jsonl"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with transcript_path.open("w", encoding="utf-8") as f:
            for event in transcript:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        messages = _transcript_to_messages(transcript, workspace=str(result.get("workspace", "")))
        first_assistant = _first_assistant_message(messages)
        negative_type = _first_negative_type(first_assistant)
        if negative_type is None:
            print(json.dumps({
                "idx": idx,
                "status": "skip",
                "reason": "first_action_not_targeted_failure",
                "run_status": result.get("status"),
            }, ensure_ascii=False))
            continue

        first_request_messages = result.get("first_request_messages") or prompt_template["messages"]
        first_request_tools = result.get("first_request_tools") or prompt_template["tools"]
        row = {
            "task_id": task.task_id,
            "variant_id": f"task_18_spreadsheet_summary-harness-focused-{idx:04d}",
            "negative_type": negative_type,
            "source": "task18_harness_runtime_student_failure",
            "model": args.model,
            "prompt_template_path": "rl/data/generated/task_18_spreadsheet_summary_runtime/runtime_prompt_template.json",
            "transcript_path": str(transcript_path),
            "tools": first_request_tools,
            "chosen": {
                "score": 1.0,
                "assistant_turns": 1,
                "messages": copy.deepcopy(first_request_messages) + [_chosen_exec_message(idx)],
                "model": "rule_exec_teacher",
            },
            "rejected": {
                "score": 0.0,
                "assistant_turns": 1,
                "messages": copy.deepcopy(first_request_messages) + [first_assistant],
                "model": args.model,
            },
            "rollout_meta": {
                "run_status": result.get("status"),
                "execution_time": result.get("execution_time"),
            },
        }
        rows.append(row)
        negative_counts[negative_type] += 1
        _write_jsonl(output_path, rows)
        print(json.dumps({
            "idx": idx,
            "status": "keep",
            "negative_type": negative_type,
            "pairs_collected": len(rows),
            "run_status": result.get("status"),
        }, ensure_ascii=False))

    summary = {
        "output": str(output_path),
        "count": len(rows),
        "target_pairs": args.target_pairs,
        "negative_counts": dict(negative_counts),
        "model": args.model,
        "vllm_base_url": args.vllm_base_url,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument(
        "--output",
        default="rl/data/generated/task_18_spreadsheet_summary_runtime/dpo_pairs_task18_harness_focused_train.jsonl",
    )
    parser.add_argument(
        "--transcript-dir",
        default="rl/data/generated/task_18_spreadsheet_summary_runtime/harness_transcripts",
    )
    parser.add_argument("--target-pairs", type=int, default=100)
    parser.add_argument("--run-id-start", type=int, default=12000)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(collect_async(args))


if __name__ == "__main__":
    main()
