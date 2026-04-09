"""
End-to-end test: vLLM (Qwen3-4B on RunPod) → ModelProxy → SSH reverse tunnel → OpenClaw (ECS) → PinchBench grading.

Usage (on RunPod):
    export PYTHONPATH="/workspace/pinchbench-skill:$PYTHONPATH"
    python3 rl/test_e2e_vllm.py --task task_00_sanity
    python3 rl/test_e2e_vllm.py --task task_02_stock
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
from pathlib import Path

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VLLM_URL = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL = "Qwen3-4B"
ECS_HOST = "8.163.82.224"
SSH_KEY = "/root/.ssh/id_ed25519"
PINCHBENCH_DIR = "/workspace/pinchbench-skill"


async def forward_to_vllm(messages: list, tools: list | None, max_tokens: int = 4096) -> dict:
    """Forward a chat completion request to the local vLLM server.
    
    Note: tools are intentionally NOT forwarded to vLLM because vLLM requires
    --enable-auto-tool-choice which adds complexity. Instead, Qwen3-4B will
    produce tool calls as text in its response, and OpenClaw will parse them.
    """
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": min(max_tokens, 4096),
        "temperature": 0.7,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            return await resp.json()


async def run_task(task_id: str, max_turns: int = 15):
    from rl.agent_loop.model_proxy import ModelProxy

    proxy = ModelProxy(host="0.0.0.0", port=0)
    port = await proxy.start()
    logger.info("ModelProxy on port %d", port)

    agent_id = f"e2e-vllm-{task_id}"
    session_id = f"e2e-{int(time.time())}"
    local_proxy_url = f"http://127.0.0.1:{port}/v1"

    models_json = json.dumps({
        "mode": "replace",
        "providers": {"verl": {
            "baseUrl": local_proxy_url, "apiKey": "dummy",
            "api": "openai-completions",
            "models": [{"id": "verl-proxy", "name": "verl-proxy"}],
        }},
        "defaultProvider": "verl", "defaultModel": "verl/verl-proxy",
    }, indent=2)
    auth_json = json.dumps({
        "version": 1,
        "profiles": {"verl-default": {"type": "api_key", "key": "dummy", "provider": "verl"}},
    }, indent=2)
    b64m = base64.b64encode(models_json.encode()).decode()
    b64a = base64.b64encode(auth_json.encode()).decode()
    ad = f"$HOME/.openclaw/agents/{agent_id}/agent"
    ws = f"/tmp/pinchbench/{task_id}"

    # Load task prompt
    task_file = Path(PINCHBENCH_DIR) / "tasks" / f"{task_id}.md"
    if not task_file.exists():
        logger.error("Task file not found: %s", task_file)
        return
    task_text = task_file.read_text()
    prompt_start = task_text.find("## Prompt")
    prompt_end = task_text.find("## Expected Behavior")
    if prompt_start < 0 or prompt_end < 0:
        prompt_start = task_text.find("## Prompt")
        prompt_end = task_text.find("## Grading")
    task_prompt = task_text[prompt_start + len("## Prompt"):prompt_end].strip()
    logger.info("Task prompt: %s...", task_prompt[:100])

    setup_cmd = " && ".join([
        f"mkdir -p {ws}",
        f"openclaw agents add {agent_id} --model verl/verl-proxy --workspace {ws} --non-interactive 2>/dev/null || true",
        f"mkdir -p {ad}",
        f"echo {b64m} | base64 -d > {ad}/models.json",
        f"echo {b64a} | base64 -d > {ad}/auth-profiles.json",
        f"rm -f $HOME/.openclaw/agents/{agent_id}/sessions/sessions.json",
    ])

    escaped_prompt = task_prompt.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    run_cmd = (
        f"openclaw agent --agent {agent_id} --session-id {session_id} "
        f'--message "{escaped_prompt}" --local'
    )

    logger.info("Launching OpenClaw on ECS with reverse tunnel...")
    t_start = time.time()
    proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "-o", "ExitOnForwardFailure=yes",
        "-R", f"{port}:127.0.0.1:{port}",
        "-i", SSH_KEY,
        f"root@{ECS_HOST}",
        f"{setup_cmd} && cd {ws} && {run_cmd}",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )

    logger.info("Waiting for OpenClaw requests (startup ~35s)...")
    turn = 0
    while turn < max_turns:
        try:
            timeout = 120 if turn == 0 else 90
            req = await asyncio.wait_for(proxy.get_request(), timeout=timeout)
            elapsed = time.time() - t_start
            logger.info(
                "Turn %d (%.1fs): %d messages, last_role=%s",
                turn, elapsed, len(req.messages), req.messages[-1].get("role", "?"),
            )

            # Forward to vLLM
            vllm_resp = await forward_to_vllm(
                req.messages, req.tools, max_tokens=req.max_tokens,
            )

            if "error" in vllm_resp:
                logger.error("vLLM error: %s", vllm_resp["error"])
                req.response_error = str(vllm_resp["error"])
                await proxy.send_response(req)
                break

            choice = vllm_resp["choices"][0]
            msg = choice["message"]
            content = msg.get("content", "")

            # Strip <think>...</think> tags for cleaner output
            import re
            clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            req.response_text = clean_content
            req.finish_reason = choice.get("finish_reason", "stop")
            req.response_usage = vllm_resp.get("usage")

            if msg.get("tool_calls"):
                req.response_tool_calls = msg["tool_calls"]

            await proxy.send_response(req)
            logger.info(
                "  Response (%d chars): %s...",
                len(clean_content), clean_content[:120],
            )
            turn += 1

        except asyncio.TimeoutError:
            elapsed = time.time() - t_start
            logger.info("No more requests after %.1fs (turn %d)", elapsed, turn)
            break

    # Wait for OpenClaw to finish
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        elapsed = time.time() - t_start
        logger.info("OpenClaw exited (code=%s, total=%.1fs, turns=%d)", proc.returncode, elapsed, turn)
        err = stderr.decode(errors="replace")
        for l in err.split("\n"):
            if any(k in l.lower() for k in ["error", "fallback"]):
                logger.warning("  STDERR: %s", l.strip()[:200])
    except asyncio.TimeoutError:
        logger.warning("OpenClaw still running, killing...")
        proc.kill()
        await proc.communicate()

    # Fetch and grade transcript
    logger.info("Fetching transcript from ECS...")
    check = await asyncio.create_subprocess_exec(
        "ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
        f"root@{ECS_HOST}",
        f"cat $HOME/.openclaw/agents/{agent_id}/sessions/{session_id}.jsonl",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await asyncio.wait_for(check.communicate(), timeout=15)
    transcript_text = out.decode(errors="replace")

    transcript = []
    for line in transcript_text.strip().split("\n"):
        try:
            transcript.append(json.loads(line))
        except Exception:
            pass

    logger.info("Transcript: %d entries", len(transcript))

    # Show assistant messages
    for entry in transcript:
        if entry.get("type") == "message":
            msg = entry.get("message", {})
            role = msg.get("role")
            content = msg.get("content", [])
            if role == "assistant" and isinstance(content, list):
                texts = [c.get("text", "")[:100] for c in content if c.get("type") == "text"]
                tools = [c.get("name", "") for c in content if c.get("type") == "toolCall"]
                logger.info("  ASSISTANT: text=%s tools=%s", texts[:2], tools[:3])

    # Run grading
    logger.info("=== GRADING ===")
    try:
        sys.path.insert(0, str(Path(PINCHBENCH_DIR) / "scripts"))
        from lib_tasks import TaskLoader
        from lib_grading import grade_task

        loader = TaskLoader(Path(PINCHBENCH_DIR) / "tasks")
        task = loader.load_task(Path(PINCHBENCH_DIR) / "tasks" / f"{task_id}.md")
        if task is None:
            logger.error("Task %s not found by loader", task_id)
        else:
            execution_result = {
                "status": "completed",
                "transcript": transcript,
                "workspace": ws,
            }
            result = grade_task(
                task=task,
                execution_result=execution_result,
                skill_dir=Path(PINCHBENCH_DIR),
                verbose=True,
            )
            logger.info("Score: %.2f / %.2f", result.score, result.max_score)
            logger.info("Breakdown: %s", result.breakdown)
            logger.info("RESULT: %s", "PASS" if result.score > 0 else "FAIL")
    except Exception as e:
        logger.error("Grading failed: %s", e, exc_info=True)

    await proxy.stop()
    logger.info("E2E TEST COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="task_00_sanity")
    parser.add_argument("--max-turns", type=int, default=15)
    args = parser.parse_args()
    asyncio.run(run_task(args.task, args.max_turns))


if __name__ == "__main__":
    main()
