"""
Debug script: test multi-turn interaction simulating the veRL agent loop path.

Mode 1 (default): Uses vLLM API for generation (tests ModelProxy + OpenClaw)
Mode 2 (VERL_SIM=1): Simulates veRL path — uses vLLM completions API to get raw
tokens, then parses tool_calls with regex like openclaw_agent_loop does.

Usage: python3 rl/debug_multi_turn.py
       VERL_SIM=1 python3 rl/debug_multi_turn.py
"""

import asyncio
import json
import logging
import os
import re
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent_loop"))
from model_proxy import ModelProxy, ModelRequest

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("debug_multi_turn")

def _ecs_host() -> str:
    h = (os.environ.get("OPENCLAW_HOST") or os.environ.get("ECS_HOST") or "").strip()
    if not h or h in ("localhost", "127.0.0.1"):
        raise SystemExit(
            "Set OPENCLAW_HOST or ECS_HOST to your ECS public IP (from cloud console; IPs change)."
        )
    return h


ECS_HOST = _ecs_host()
ECS_PORT = int(os.environ.get("ECS_PORT", "22"))
ECS_USER = os.environ.get("ECS_USER", "root")
SSH_KEY = os.environ.get("SSH_KEY", "/root/.ssh/id_ed25519")
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
TASK_ID = os.environ.get("TASK_ID", "cli_ls_hidden")
WORKSPACE_BASE = os.environ.get("WORKSPACE_BASE", "/tmp/pinchbench-debug")
MAX_TURNS = 8
VERL_SIM = os.environ.get("VERL_SIM", "0") == "1"


def parse_tool_calls(text):
    """Same regex as openclaw_agent_loop._parse_tool_calls"""
    tool_calls = []
    for match in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
        try:
            tc = json.loads(match.group(1))
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                },
            })
        except json.JSONDecodeError:
            pass
    return tool_calls if tool_calls else None


def strip_tool_tags(text):
    """Same as openclaw_agent_loop._strip_tool_tags"""
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


async def call_vllm_chat(messages, tools=None):
    """Call vLLM chat completions API (mode 1)."""
    import aiohttp
    payload = {
        "model": "Qwen3-4B",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()
            if "choices" not in data:
                logger.error("vLLM error: %s", json.dumps(data, ensure_ascii=False)[:500])
                raise RuntimeError(f"vLLM error: {data}")
            return data["choices"][0]


def prepare_messages(messages, tools):
    """Same as openclaw_agent_loop._prepare_messages"""
    out = []
    for msg in messages:
        msg = dict(msg)
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            msg["content"] = "\n".join(parts)
        out.append(msg)

    if tools and os.environ.get("PINCHBENCH_RL_INJECT_TOOL_FORMAT_SUFFIX", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        TOOL_FORMAT_SUFFIX = (
            "\n\n# Output Format for Tool Calls\n"
            "When you need to call a tool, output the call inside <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": "<function-name>", "arguments": {<args-json-object>}}\n'
            "</tool_call>\n"
            "You may call multiple tools by using multiple <tool_call> blocks."
        )
        for msg in out:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str) and "<tool_call>" not in content:
                    msg["content"] = content + TOOL_FORMAT_SUFFIX
                break
    return out


async def call_vllm_verl_sim(messages, tools=None):
    """Simulate veRL path: inject tool format + apply_chat_template (no tools param) + raw generate + regex parse."""
    from transformers import AutoTokenizer
    MODEL_PATH = "/workspace/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

    if not hasattr(call_vllm_verl_sim, "_tok"):
        call_vllm_verl_sim._tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok = call_vllm_verl_sim._tok

    messages = prepare_messages(messages, tools)
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    logger.info("[verl_sim] prompt_ids=%d", len(prompt_ids))

    import aiohttp
    payload = {
        "model": "Qwen3-4B",
        "prompt": prompt_text,
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{VLLM_URL}/v1/completions", json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()
            if "choices" not in data:
                logger.error("vLLM error: %s", json.dumps(data, ensure_ascii=False)[:500])
                raise RuntimeError(f"vLLM error: {data}")
            raw_text = data["choices"][0]["text"]

    logger.info("[verl_sim] raw_text: %s", repr(raw_text[:300]))
    tc = parse_tool_calls(raw_text)
    content = strip_tool_tags(raw_text)

    return {
        "message": {
            "role": "assistant",
            "content": content,
            "tool_calls": tc,
        },
        "finish_reason": "tool_calls" if tc else "stop",
    }


async def main():
    logger.info("Mode: %s", "verl_sim" if VERL_SIM else "vllm_chat")

    proxy = ModelProxy(host="0.0.0.0", port=0, timeout=120)
    proxy_port = await proxy.start()
    logger.info("ModelProxy listening on port %d", proxy_port)

    task_prompt = "List all files (including hidden files) in the current directory."

    agent_id = f"debug-{uuid.uuid4().hex[:8]}"
    workspace = f"{WORKSPACE_BASE}/{TASK_ID}"
    local_proxy_url = f"http://127.0.0.1:{proxy_port}/v1"

    import base64
    models_json = json.dumps({
        "mode": "replace",
        "providers": {"verl": {
            "baseUrl": local_proxy_url, "apiKey": "dummy", "api": "openai-completions",
            "models": [{"id": "verl-proxy", "name": "verl-proxy"}],
        }},
        "defaultProvider": "verl", "defaultModel": "verl/verl-proxy",
    }, indent=2)
    auth_json = json.dumps({
        "version": 1,
        "profiles": {
            "verl-default": {
                "type": "api_key",
                "key": "dummy",
                "provider": "verl",
            }
        },
    }, indent=2)
    agent_dir = f"$HOME/.openclaw/agents/{agent_id}/agent"
    b64_models = base64.b64encode(models_json.encode()).decode()
    b64_auth = base64.b64encode(auth_json.encode()).decode()

    activate_cmd = os.environ.get("OPENCLAW_REMOTE_ACTIVATE_CMD", "").strip()
    setup_parts = [f"mkdir -p {workspace}"]
    if activate_cmd:
        setup_parts.append(activate_cmd)
    setup_parts.extend([
        f"openclaw agents add {agent_id} --model verl/verl-proxy --workspace {workspace} --non-interactive 2>/dev/null || true",
        f"mkdir -p {agent_dir}",
        f"echo {b64_models} | base64 -d > {agent_dir}/models.json",
        f"echo {b64_auth} | base64 -d > {agent_dir}/auth-profiles.json",
        f"rm -f $HOME/.openclaw/agents/{agent_id}/sessions/sessions.json",
    ])
    setup_cmd = " && ".join(setup_parts)

    escaped_prompt = task_prompt.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    run_cmd = (
        f"openclaw agent --agent {agent_id} --session-id debug-{uuid.uuid4().hex[:8]} "
        f'--message "{escaped_prompt}" --local'
    )

    full_cmd = f"{setup_cmd} && cd {workspace} && {run_cmd}"

    logger.info("Starting OpenClaw on ECS with SSH reverse tunnel (port %d)...", proxy_port)
    proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "-o", "ExitOnForwardFailure=yes",
        "-R", f"{proxy_port}:127.0.0.1:{proxy_port}",
        "-i", SSH_KEY, "-p", str(ECS_PORT),
        f"{ECS_USER}@{ECS_HOST}",
        full_cmd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    logger.info("OpenClaw SSH process pid=%d", proc.pid)

    async def drain_stream(stream, label):
        while True:
            line = await stream.readline()
            if not line:
                break
            logger.info("[ECS %s] %s", label, line.decode(errors="replace").rstrip())

    asyncio.create_task(drain_stream(proc.stdout, "stdout"))
    asyncio.create_task(drain_stream(proc.stderr, "stderr"))

    turn_count = 0
    try:
        while turn_count < MAX_TURNS:
            logger.info("=== Waiting for turn %d request ===", turn_count)
            try:
                req: ModelRequest = await asyncio.wait_for(proxy.get_request(), timeout=120)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for turn %d", turn_count)
                break

            if proc.returncode is not None:
                logger.info("OpenClaw exited (code=%s) before turn %d", proc.returncode, turn_count)
                break

            logger.info("Turn %d: got %d messages, tools=%d", turn_count, len(req.messages), len(req.tools or []))
            for i, m in enumerate(req.messages):
                role = m.get("role", "?")
                content = (m.get("content") or "")[:200] if isinstance(m.get("content"), str) else str(m.get("content", ""))[:200]
                tc = m.get("tool_calls", [])
                logger.info("  msg[%d] role=%s content=%s... tool_calls=%d", i, role, content[:80], len(tc) if tc else 0)

            if VERL_SIM:
                choice = await call_vllm_verl_sim(req.messages, req.tools)
            else:
                choice = await call_vllm_chat(req.messages, req.tools)

            msg = choice["message"]
            finish = choice.get("finish_reason", "stop")

            logger.info("Turn %d response: finish_reason=%s", turn_count, finish)
            logger.info("  content: %s", ((msg.get("content") or "")[:200]))
            logger.info("  tool_calls: %s", json.dumps(msg.get("tool_calls") or [], ensure_ascii=False)[:300])

            content_text = msg.get("content") or ""
            if msg.get("tool_calls"):
                req.response_text = content_text
                req.response_tool_calls = msg["tool_calls"]
                req.finish_reason = "tool_calls"
            else:
                req.response_text = content_text
                req.response_tool_calls = None
                req.finish_reason = "stop"

            logger.info("Turn %d: sending to OpenClaw (finish=%s, tc=%s)",
                        turn_count, req.finish_reason, bool(req.response_tool_calls))
            await proxy.send_response(req)
            turn_count += 1

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        await proxy.drain()
        await proxy.stop()

    logger.info("Done. Completed %d turns.", turn_count)


if __name__ == "__main__":
    asyncio.run(main())
