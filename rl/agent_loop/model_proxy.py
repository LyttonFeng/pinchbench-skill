"""
ModelProxy: HTTP reverse proxy that intercepts OpenClaw's LLM requests
and forwards them to veRL's vLLM inference engine.

Architecture:
  OpenClaw agent → POST /v1/chat/completions → ModelProxy (aiohttp)
  ModelProxy → asyncio.Queue → OpenClawAgentLoop → veRL server_manager.generate()
  veRL response → ModelProxy → OpenAI-format SSE stream → OpenClaw agent

Based on veRL's SWE-Agent recipe ModelProxy pattern.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from aiohttp import web

logger = logging.getLogger(__name__)


@dataclass
class ModelRequest:
    """A single LLM request from OpenClaw, waiting for veRL to generate."""

    request_id: str
    messages: list[dict[str, Any]]
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None
    received_at: float = field(default_factory=time.time)
    response_event: asyncio.Event = field(default_factory=asyncio.Event)
    response_text: Optional[str] = None
    response_tool_calls: Optional[list[dict]] = None
    response_error: Optional[str] = None
    response_usage: Optional[dict] = None
    finish_reason: str = "stop"


class ModelProxy:
    """HTTP proxy server that intercepts OpenClaw's LLM calls.

    Supports both streaming (SSE) and non-streaming responses.
    OpenClaw always sends stream=true, so streaming is the primary path.

    Lifecycle:
        proxy = ModelProxy()
        await proxy.start()       # binds to ephemeral port
        ...                       # OpenClaw sends requests to proxy.port
        req = await proxy.get_request()    # dequeue one request
        await proxy.send_response(req)     # unblock the HTTP handler
        ...
        await proxy.stop()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0, timeout: float = 600.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._queue: asyncio.Queue[ModelRequest] = asyncio.Queue()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    async def start(self) -> int:
        self._app = web.Application()
        self._app.router.add_post("/v1/chat/completions", self._handle_chat_completion)
        self._app.router.add_get("/v1/models", self._handle_list_models)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        sockets = self._site._server.sockets  # type: ignore[union-attr]
        if sockets:
            self.port = sockets[0].getsockname()[1]

        logger.info("ModelProxy listening on %s:%d", self.host, self.port)
        return self.port

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
        logger.info("ModelProxy stopped")

    async def get_request(self, timeout: Optional[float] = None) -> ModelRequest:
        t = timeout or self.timeout
        return await asyncio.wait_for(self._queue.get(), timeout=t)

    async def send_response(self, req: ModelRequest) -> None:
        req.response_event.set()

    async def drain(self) -> None:
        """Drain any pending requests with error responses."""
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                req.response_error = "proxy shutting down"
                req.response_event.set()
            except asyncio.QueueEmpty:
                break

    # ── HTTP handlers ──

    async def _handle_chat_completion(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"error": {"message": "invalid JSON"}}, status=400
            )

        req = ModelRequest(
            request_id=f"proxy-{uuid.uuid4().hex[:12]}",
            messages=body.get("messages", []),
            temperature=body.get("temperature", 0.7),
            max_tokens=body.get("max_tokens", body.get("max_completion_tokens", 4096)),
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
        )

        await self._queue.put(req)
        logger.info(
            "Queued request %s (%d messages, stream=%s, tools=%d)",
            req.request_id, len(req.messages), body.get("stream", False),
            len(body.get("tools") or []),
        )

        try:
            await asyncio.wait_for(req.response_event.wait(), timeout=self.timeout)
        except asyncio.TimeoutError:
            return web.json_response(
                {"error": {"message": "generation timeout"}}, status=504
            )

        if req.response_error:
            return web.json_response(
                {"error": {"message": req.response_error}}, status=500
            )

        is_stream = body.get("stream", False)
        model_name = body.get("model", "verl-proxy")

        logger.info(
            "Responding to %s: stream=%s, tool_calls=%d, content_len=%d, finish=%s",
            req.request_id, is_stream,
            len(req.response_tool_calls or []),
            len(req.response_text or ""),
            req.finish_reason,
        )

        if is_stream:
            return await self._stream_response(request, req, model_name)
        return self._json_response(req, model_name)

    def _json_response(self, req: ModelRequest, model: str) -> web.Response:
        message: dict[str, Any] = {"role": "assistant", "content": req.response_text or ""}
        if req.response_tool_calls:
            message["tool_calls"] = req.response_tool_calls

        body = {
            "id": req.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": req.finish_reason}],
            "usage": req.response_usage or {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            },
        }
        return web.json_response(body)

    async def _stream_response(
        self, http_request: web.Request, req: ModelRequest, model: str,
    ) -> web.StreamResponse:
        """Return an SSE stream matching the OpenAI chat.completions streaming format.

        Follows the exact OpenAI streaming spec:
        - When tool_calls present: role chunk -> tool_call chunks (name+args split) -> finish(tool_calls)
        - When no tool_calls: role chunk -> content chunk -> finish(stop)
        - tool_calls and content are never mixed in the same delta
        """
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await resp.prepare(http_request)

        created = int(time.time())
        has_tool_calls = bool(req.response_tool_calls)

        async def _send(data: dict) -> None:
            payload = f"data: {json.dumps(data)}\n\n"
            logger.debug("[SSE %s] %s", req.request_id, payload.strip()[:300])
            await resp.write(payload.encode())

        def _chunk(delta: dict, finish_reason: str | None = None) -> dict:
            c: dict = {
                "id": req.request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return c

        # Chunk 1: role
        await _send(_chunk({"role": "assistant"}))

        if has_tool_calls:
            for i, tc in enumerate(req.response_tool_calls):
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                # Send each tool call as a single chunk with all fields.
                # OpenClaw's streaming parser starts a new toolCall block
                # when it sees a chunk with an id.
                await _send(_chunk({"tool_calls": [{
                    "index": i,
                    "id": tc.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args_str,
                    },
                }]}))

            await _send(_chunk({}, finish_reason="tool_calls"))
        else:
            content = req.response_text or ""
            if content:
                await _send(_chunk({"content": content}))
            await _send(_chunk({}, finish_reason="stop"))

        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def _handle_list_models(self, request: web.Request) -> web.Response:
        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": "verl-proxy",
                    "object": "model",
                    "owned_by": "verl",
                }
            ],
        })

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})
