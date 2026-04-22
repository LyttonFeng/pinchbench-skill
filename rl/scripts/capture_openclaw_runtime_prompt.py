#!/usr/bin/env python3
"""Capture the first OpenClaw -> /v1/chat/completions request for a task.

Uses a tiny stdlib HTTP server, so it does not depend on aiohttp or veRL.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_agent import cleanup_agent_sessions, ensure_agent_exists, slugify_model  # noqa: E402
from lib_tasks import TaskLoader, resolve_task_markdown_path  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class CaptureState:
    def __init__(self) -> None:
        self.request_body: dict[str, Any] | None = None
        self.event = threading.Event()


def _make_handler(state: CaptureState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/models":
                data = {
                    "object": "list",
                    "data": [{"id": "Qwen3-1.7B", "object": "model", "owned_by": "capture"}],
                }
                body = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/health":
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                state.request_body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                state.request_body = {"_raw": raw.decode("utf-8", errors="replace")}
            state.event.set()

            stream = False
            if isinstance(state.request_body, dict):
                stream = bool(state.request_body.get("stream", False))
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                chunk = {
                    "id": "capture",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "Qwen3-1.7B",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                final = {
                    "id": "capture",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "Qwen3-1.7B",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                self.wfile.write(f"data: {json.dumps(final)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                return

            body = {
                "id": "capture",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "Qwen3-1.7B",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def _load_task(task_id: str):
    loader = TaskLoader(REPO_ROOT / "tasks")
    path = resolve_task_markdown_path(REPO_ROOT / "tasks", task_id)
    return loader.load_task(path)


def _prepare_workspace(task, run_root: Path) -> Path:
    workspace = run_root / "agent_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    assets_dir = REPO_ROOT / "assets"
    for spec in task.workspace_files:
        src = assets_dir / spec["source"]
        dst = workspace / spec["dest"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return workspace


def _extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return ""
    first = messages[0]
    if first.get("role") != "system":
        return ""
    content = first.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="task_18_spreadsheet_summary")
    parser.add_argument("--model-id", default="Qwen3-1.7B")
    parser.add_argument(
        "--output",
        default="rl/data/generated/task_18_spreadsheet_summary_runtime/runtime_prompt_template.json",
    )
    args = parser.parse_args()

    task = _load_task(args.task_id)
    run_root = REPO_ROOT / "rl" / "data" / "generated" / f"{args.task_id}_runtime_capture"
    workspace = _prepare_workspace(task, run_root)
    agent_id = f"capture-{args.task_id}-{slugify_model(args.model_id)}"

    state = CaptureState()
    server = _ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}/v1"
    logger.info("Capture server listening on %s", base_url)

    os.environ["PINCHBENCH_FORCE_LOCAL_OPENCLAW"] = "1"
    cleanup_agent_sessions(agent_id)
    ensure_agent_exists(
        agent_id=agent_id,
        model_id=args.model_id,
        workspace_dir=workspace,
        base_url=base_url,
        api_key="dummy",
    )

    cmd = [
        "openclaw",
        "agent",
        "--agent",
        agent_id,
        "--session-id",
        f"capture-{int(time.time() * 1000)}",
        "--message",
        task.prompt,
        "--local",
    ]
    logger.info("Launching OpenClaw task %s", args.task_id)
    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        if not state.event.wait(timeout=90):
            raise TimeoutError("Timed out waiting for OpenClaw runtime request")
        body = state.request_body or {}
        messages = body.get("messages", []) if isinstance(body, dict) else []
        payload = {
            "task_id": args.task_id,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": "openclaw_runtime_first_request",
            "model_id": args.model_id,
            "tool_choice": body.get("tool_choice") if isinstance(body, dict) else None,
            "tool_count": len(body.get("tools", []) or []) if isinstance(body, dict) else 0,
            "system_prompt": _extract_system_prompt(messages),
            "messages": messages,
            "tools": body.get("tools", []) if isinstance(body, dict) else [],
            "raw_request": body,
            "notes": {
                "workspace": str(workspace),
                "message_count": len(messages),
                "captures_first_request_only": True,
            },
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info("Wrote runtime template to %s", output)
    finally:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
