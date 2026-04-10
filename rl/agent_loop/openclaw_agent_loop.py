"""
OpenClawAgentLoop: veRL agent loop that drives OpenClaw as the RL environment.

Architecture:
  1. Inherit from veRL's AgentLoopBase
  2. Start ModelProxy on ephemeral port
  3. SSH to ECS, run `openclaw agent --local` pointing LLM at ModelProxy
  4. Intercept each LLM request, use self.server_manager.generate() for token generation
  5. Convert tokens back to text, return via ModelProxy to OpenClaw
  6. After episode, run grading and compute process rewards
  7. Return AgentLoopOutput for veRL training

Environment variables:
  OPENCLAW_HOST, OPENCLAW_USER, OPENCLAW_SSH_KEY, OPENCLAW_PORT,
  OPENCLAW_WORKSPACE, PINCHBENCH_DIR, REWARD_MODE,
  PRM_VLLM_BASE_URL, PRM_MODEL, PRM_API_KEY
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    logger.addHandler(_h)


@dataclass
class OpenClawConfig:
    host: str = "localhost"
    user: str = "root"
    ssh_key: str = str(Path.home() / ".ssh" / "id_ed25519")
    ssh_port: int = 22
    workspace_base: str = "/tmp/pinchbench"
    pinchbench_dir: str = ""
    reward_mode: str = "self-judge"
    proxy_bind_host: str = "0.0.0.0"
    agent_timeout: float = 120.0
    max_turns: int = 5
    prm_vllm_base_url: str = "http://localhost:9090/v1"
    prm_model: str = "Qwen/Qwen3-4B"
    prm_api_key: str = "dummy"

    @classmethod
    def from_env(cls) -> "OpenClawConfig":
        return cls(
            host=os.environ.get("OPENCLAW_HOST", "localhost"),
            user=os.environ.get("OPENCLAW_USER", "root"),
            ssh_key=os.environ.get("OPENCLAW_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519")),
            ssh_port=int(os.environ.get("OPENCLAW_PORT", "22")),
            workspace_base=os.environ.get("OPENCLAW_WORKSPACE", "/tmp/pinchbench"),
            pinchbench_dir=os.environ.get("PINCHBENCH_DIR", ""),
            reward_mode=os.environ.get("REWARD_MODE", "self-judge"),
            proxy_bind_host=os.environ.get("PROXY_BIND_HOST", "0.0.0.0"),
            agent_timeout=float(os.environ.get("AGENT_TIMEOUT", "120")),
            max_turns=int(os.environ.get("MAX_TURNS", "5")),
            prm_vllm_base_url=os.environ.get("PRM_VLLM_BASE_URL", "http://localhost:9090/v1"),
            prm_model=os.environ.get("PRM_MODEL", "Qwen/Qwen3-4B"),
            prm_api_key=os.environ.get("PRM_API_KEY", "dummy"),
        )


class OpenClawAgentLoop(AgentLoopBase):
    """veRL agent loop for PinchBench via OpenClaw."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        if config and isinstance(config, dict):
            self.oc_config = OpenClawConfig(**config)
        else:
            self.oc_config = OpenClawConfig.from_env()

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run one PinchBench episode.

        kwargs contains dataset fields: raw_prompt, extra_info, etc.
        """
        from .model_proxy import ModelProxy, ModelRequest
        from .trajectory import TrajectoryReconstructor, TurnRecord

        print(f"[OpenClawAgentLoop.run] START kwargs keys={list(kwargs.keys())}", flush=True)
        logger.info("[run] START kwargs keys=%s, sampling_params=%s", list(kwargs.keys()), sampling_params)

        raw_prompt = kwargs.get("raw_prompt", [])
        extra_info = kwargs.get("extra_info", {})
        task_id = extra_info.get("task_id", "unknown")
        task_prompt = ""
        if raw_prompt:
            last_user = [m for m in raw_prompt if m.get("role") == "user"]
            if last_user:
                task_prompt = last_user[-1].get("content", "")

        print(f"[OpenClawAgentLoop.run] task_id={task_id}, host={self.oc_config.host}, prompt_len={len(task_prompt)}", flush=True)
        logger.info("[run] task_id=%s, host=%s, prompt_len=%d", task_id, self.oc_config.host, len(task_prompt))

        session_id = f"rl-{uuid.uuid4().hex[:8]}"

        proxy = ModelProxy(host=self.oc_config.proxy_bind_host, port=0)
        proxy_port = await proxy.start()
        print(f"[OpenClawAgentLoop.run] ModelProxy started on port {proxy_port}", flush=True)
        logger.info("[run] ModelProxy started on port %d", proxy_port)

        all_prompt_ids: list[int] = []
        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        all_response_logprobs: list[float] = []
        turns: list[TurnRecord] = []
        messages: list[dict] = []
        turn_count = 0
        retry_count = 0

        t_start = time.time()
        openclaw_proc: Optional[asyncio.subprocess.Process] = None
        metrics: dict[str, Any] = {}

        try:
            if self.oc_config.host in ("localhost", "127.0.0.1"):
                logger.info("[run] Starting local OpenClaw for task=%s", task_id)
                openclaw_proc = await self._start_local_openclaw(
                    task_prompt, session_id, f"http://127.0.0.1:{proxy_port}/v1", task_id,
                )
            else:
                logger.info("[run] Starting remote OpenClaw on %s for task=%s", self.oc_config.host, task_id)
                openclaw_proc = await self._start_remote_openclaw(
                    task_prompt, session_id, proxy_port, task_id,
                )
            logger.info("[run] OpenClaw started pid=%s, waiting for first request...", openclaw_proc.pid)

            async def _drain_oc_stream(stream, label, tid):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    logger.debug("[OC %s/%s] %s", tid, label, line.decode(errors="replace").rstrip())

            asyncio.get_event_loop().create_task(_drain_oc_stream(openclaw_proc.stdout, "out", task_id))
            asyncio.get_event_loop().create_task(_drain_oc_stream(openclaw_proc.stderr, "err", task_id))

            while turn_count < self.oc_config.max_turns:
                try:
                    logger.info("[run] Waiting for proxy request (turn=%d, timeout=%.0fs)...", turn_count, self.oc_config.agent_timeout)
                    proxy_task = asyncio.ensure_future(proxy.get_request())
                    proc_task = asyncio.ensure_future(openclaw_proc.wait())
                    done, pending = await asyncio.wait(
                        [proxy_task, proc_task],
                        timeout=self.oc_config.agent_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for p in pending:
                        p.cancel()

                    if not done:
                        logger.warning("Proxy timeout at turn %d", turn_count)
                        break

                    if proc_task in done and proxy_task not in done:
                        logger.info("OpenClaw exited (code=%s) before sending request", openclaw_proc.returncode)
                        break

                    req: ModelRequest = proxy_task.result()
                    logger.info("[run] Got proxy request turn=%d messages=%d", turn_count, len(req.messages))
                except asyncio.CancelledError:
                    logger.warning("Cancelled at turn %d", turn_count)
                    break

                if openclaw_proc.returncode is not None:
                    logger.info("OpenClaw exited (code=%s)", openclaw_proc.returncode)
                    req.response_error = "agent process exited"
                    await proxy.send_response(req)
                    break

                n_msg = len(req.messages)
                if turn_count > 0 and n_msg <= 2 and retry_count >= 2:
                    logger.warning("OpenClaw stuck in retry loop (turn=%d, retries=%d), breaking", turn_count, retry_count)
                    req.response_error = "retry loop detected"
                    await proxy.send_response(req)
                    break
                if turn_count > 0 and n_msg <= 2:
                    retry_count += 1
                else:
                    retry_count = 0

                chat_messages = self._prepare_messages(req.messages, req.tools)
                try:
                    logger.info("[run] Applying chat template (messages=%d, tools=%s)...", len(chat_messages), bool(req.tools))
                    prompt_token_ids = await self.apply_chat_template(chat_messages)
                    logger.info("[run] Chat template done, prompt_ids=%d", len(prompt_token_ids))
                except Exception as e:
                    logger.error("Chat template failed: %s", e)
                    req.response_error = str(e)
                    await proxy.send_response(req)
                    break

                if turn_count == 0:
                    all_prompt_ids = list(prompt_token_ids)

                logger.info("[run] Calling server_manager.generate (turn=%d, prompt_ids=%d)...", turn_count, len(prompt_token_ids))
                gen_output = await self.server_manager.generate(
                    request_id=uuid.uuid4().hex,
                    prompt_ids=prompt_token_ids,
                    sampling_params=sampling_params,
                )
                logger.info("[run] Generate done, got %d tokens", len(gen_output.token_ids) if gen_output.token_ids else 0)

                response_ids = list(gen_output.token_ids)
                response_logprobs = list(gen_output.log_probs) if gen_output.log_probs else [0.0] * len(response_ids)

                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                clean_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                # Env tokens = everything added by environment between turns
                if turn_count > 0:
                    env_token_ids = prompt_token_ids[len(all_prompt_ids) + len(all_response_ids):]
                    all_response_ids.extend(env_token_ids)
                    all_response_mask.extend([0] * len(env_token_ids))
                    all_response_logprobs.extend([0.0] * len(env_token_ids))

                all_response_ids.extend(response_ids)
                all_response_mask.extend([1] * len(response_ids))
                all_response_logprobs.extend(response_logprobs)

                turns.append(TurnRecord(
                    turn_index=turn_count,
                    messages=chat_messages,
                    prompt_ids=prompt_token_ids,
                    response_ids=response_ids,
                    response_text=clean_text,
                    response_logprobs=response_logprobs,
                ))

                tool_calls = self._parse_tool_calls(response_text)

                content_for_openclaw = self._strip_tool_tags(clean_text)
                logger.info("[run] Turn %d: tool_calls=%s, finish=%s, content_len=%d",
                            turn_count, bool(tool_calls),
                            "tool_calls" if tool_calls else "stop",
                            len(content_for_openclaw))
                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        f = tc.get("function", {})
                        logger.info("[run]   tc[%d] id=%s name=%s args_len=%d args_preview=%.200s",
                                    i, tc.get("id"), f.get("name"), len(f.get("arguments", "")),
                                    f.get("arguments", ""))
                logger.info("[run] raw_text_preview=%.300s", response_text[:300] if response_text else "")

                req.response_text = content_for_openclaw
                req.response_tool_calls = tool_calls
                req.finish_reason = "tool_calls" if tool_calls else "stop"
                await proxy.send_response(req)

                messages.append({"role": "assistant", "content": clean_text, "tool_calls": tool_calls or []})
                turn_count += 1

                total_resp_len = len(all_response_ids)
                resp_budget = self.rollout_config.response_length
                logger.info("[run] Accumulated response tokens: %d / %d (%.0f%%)",
                            total_resp_len, resp_budget, 100 * total_resp_len / resp_budget if resp_budget else 0)
                if total_resp_len >= resp_budget * 0.85:
                    logger.warning("Response token budget nearly exhausted (%d/%d), stopping early", total_resp_len, resp_budget)
                    break

            if openclaw_proc and openclaw_proc.returncode is None:
                try:
                    await asyncio.wait_for(openclaw_proc.wait(), timeout=30)
                except asyncio.TimeoutError:
                    openclaw_proc.kill()
                    await openclaw_proc.wait()

        finally:
            await proxy.drain()
            await proxy.stop()

        t_gen = time.time() - t_start
        metrics["generate_sequences"] = t_gen
        metrics["num_preempted"] = -1

        # Compute reward
        transcript_raw = self._load_openclaw_transcript(session_id, task_id)
        terminal_success = self._run_grading(task_id, transcript_raw)
        terminal_reward = 1.0 if terminal_success else -1.0

        trajectory_for_reward = self._transcript_to_messages(transcript_raw) or messages
        per_turn_rewards = await self._compute_rewards(
            trajectory_for_reward, terminal_success, task_id, task_prompt,
        )
        # per_turn_rewards already includes terminal_reward on the last turn
        # (added by compute_episode_rewards_async), so do NOT add it again
        total_reward = sum(per_turn_rewards)
        logger.info("Reward: total=%.2f, terminal=%.1f, per_turn=%s, turns=%d, mode=%s",
                     total_reward, terminal_reward, per_turn_rewards, turn_count, self.oc_config.reward_mode)

        # Assign per-turn rewards at <|im_end|> token positions
        # terminal_reward=0 here because it's already in per_turn_rewards
        reward_at_tokens = self._assign_rewards(
            all_response_ids, all_response_mask, per_turn_rewards, 0.0,
        )

        response_length = self.rollout_config.response_length

        # Ensure non-empty ids for veRL postprocessing (tokenizer.pad needs valid input)
        if not all_prompt_ids:
            eos_id = self.tokenizer.eos_token_id or 0
            all_prompt_ids = [eos_id]
        if not all_response_ids:
            eos_id = self.tokenizer.eos_token_id or 0
            all_response_ids = [eos_id]
            all_response_mask = [0]
            all_response_logprobs = [0.0]

        output = AgentLoopOutput(
            prompt_ids=all_prompt_ids,
            response_ids=all_response_ids[:response_length],
            response_mask=all_response_mask[:response_length],
            response_logprobs=all_response_logprobs[:response_length] if all_response_logprobs else None,
            reward_score=total_reward,
            num_turns=turn_count * 2 + 1,
            metrics=AgentLoopMetrics(
                generate_sequences=t_gen,
                tool_calls=turn_count,
                num_preempted=-1,
            ),
            extra_fields={
                "turn_scores": per_turn_rewards,
                "tool_rewards": reward_at_tokens[:response_length],
                "task_id": task_id,
                "terminal_success": terminal_success,
                "terminal_reward": terminal_reward,
                "reward_mode": self.oc_config.reward_mode,
                "trajectory": trajectory_for_reward,
            },
        )
        return output

    # ── OpenClaw process management ──

    async def _start_local_openclaw(
        self, prompt: str, session_id: str, proxy_url: str, task_id: str,
    ) -> asyncio.subprocess.Process:
        agent_id = f"rl-{task_id}-{session_id[:8]}"
        workspace = Path(self.oc_config.workspace_base) / task_id
        self._setup_agent_local(agent_id, proxy_url, workspace)
        self._prepare_workspace(task_id, workspace)

        proc = await asyncio.create_subprocess_exec(
            "openclaw", "agent", "--agent", agent_id,
            "--session-id", session_id, "--message", prompt, "--local",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )
        logger.info("Started local OpenClaw pid=%s task=%s", proc.pid, task_id)
        return proc

    async def _start_remote_openclaw(
        self, prompt: str, session_id: str, proxy_port: int, task_id: str,
    ) -> asyncio.subprocess.Process:
        agent_id = f"rl-{task_id}-{session_id[:8]}"
        workspace = f"{self.oc_config.workspace_base}/{task_id}"

        # SSH reverse tunnel: ECS localhost:<proxy_port> -> RunPod localhost:<proxy_port>
        # RunPod doesn't expose arbitrary ports to the public internet,
        # so OpenClaw on ECS connects to its own localhost via the tunnel.
        local_proxy_url = f"http://127.0.0.1:{proxy_port}/v1"

        setup_cmd = self._build_remote_setup(agent_id, local_proxy_url, workspace, task_id)
        escaped_prompt = prompt.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
        run_cmd = (
            f"openclaw agent --agent {agent_id} --session-id {session_id} "
            f'--message "{escaped_prompt}" --local'
        )

        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            "-o", "ExitOnForwardFailure=yes",
            "-R", f"{proxy_port}:127.0.0.1:{proxy_port}",
            "-i", self.oc_config.ssh_key, "-p", str(self.oc_config.ssh_port),
            f"{self.oc_config.user}@{self.oc_config.host}",
            f"{setup_cmd} && cd {workspace} && {run_cmd}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        logger.info(
            "Started remote OpenClaw on %s pid=%s task=%s (reverse tunnel :%d)",
            self.oc_config.host, proc.pid, task_id, proxy_port,
        )
        return proc

    def _setup_agent_local(self, agent_id: str, proxy_url: str, workspace: Path) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["openclaw", "agents", "add", agent_id,
             "--model", "verl/verl-proxy", "--workspace", str(workspace), "--non-interactive"],
            capture_output=True, text=True, check=False,
        )
        agent_dir = Path.home() / ".openclaw" / "agents" / agent_id / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        models = {
            "mode": "replace",
            "providers": {"verl": {
                "baseUrl": proxy_url, "apiKey": "dummy", "api": "openai-completions",
                "models": [{"id": "verl-proxy", "name": "verl-proxy"}],
            }},
            "defaultProvider": "verl", "defaultModel": "verl/verl-proxy",
        }
        (agent_dir / "models.json").write_text(json.dumps(models, indent=2), "utf-8")
        auth = {
            "version": 1,
            "profiles": {
                "verl-default": {
                    "type": "api_key",
                    "key": "dummy",
                    "provider": "verl",
                }
            },
        }
        (agent_dir / "auth-profiles.json").write_text(json.dumps(auth, indent=2), "utf-8")
        sessions_file = Path.home() / ".openclaw" / "agents" / agent_id / "sessions" / "sessions.json"
        if sessions_file.exists():
            sessions_file.unlink()

    def _build_remote_setup(self, agent_id: str, proxy_url: str, workspace: str, task_id: str) -> str:
        import base64
        models_json = json.dumps({
            "mode": "replace",
            "providers": {"verl": {
                "baseUrl": proxy_url, "apiKey": "dummy", "api": "openai-completions",
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
        lock_file = "/tmp/openclaw_agents.lock"
        return " && ".join([
            f"mkdir -p {workspace}",
            f"(flock {lock_file} openclaw agents add {agent_id} --model verl/verl-proxy --workspace {workspace} --non-interactive 2>&1 || echo '[WARN] openclaw agents add failed for {agent_id}' >&2)",
            f"mkdir -p {agent_dir}",
            f"echo {b64_models} | base64 -d > {agent_dir}/models.json",
            f"echo {b64_auth} | base64 -d > {agent_dir}/auth-profiles.json",
            f"rm -f $HOME/.openclaw/agents/{agent_id}/sessions/sessions.json",
        ])

    def _prepare_workspace(self, task_id: str, workspace: Path) -> None:
        if not self.oc_config.pinchbench_dir:
            return
        try:
            import sys, shutil
            scripts_dir = Path(self.oc_config.pinchbench_dir) / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from lib_tasks import TaskLoader
            tasks_dir = Path(self.oc_config.pinchbench_dir) / "tasks"
            loader = TaskLoader(tasks_dir)
            task_file = tasks_dir / f"{task_id}.md"
            if not task_file.exists():
                return
            task = loader.load_task(task_file)
            if task is None:
                return
            if workspace.exists():
                shutil.rmtree(workspace)
            workspace.mkdir(parents=True, exist_ok=True)
            for f in getattr(task, "workspace_files", []):
                if "content" in f:
                    dest = workspace / f["path"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(f["content"])
                elif "source" in f:
                    src = Path(self.oc_config.pinchbench_dir) / "assets" / f["source"]
                    dest = workspace / f.get("dest", f["source"])
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(src.read_bytes())
        except Exception as e:
            logger.error("Workspace prep failed: %s", e)

    # ── Transcript / Grading ──

    def _load_openclaw_transcript(self, session_id: str, task_id: str) -> list[dict]:
        agents_dir = Path.home() / ".openclaw" / "agents"
        if not agents_dir.exists():
            return []
        prefix = f"rl-{task_id}"
        for agent_dir in agents_dir.iterdir():
            if not agent_dir.name.startswith(prefix):
                continue
            sessions_dir = agent_dir / "sessions"
            if not sessions_dir.exists():
                continue
            jsonl_files = sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if jsonl_files:
                transcript = []
                for line in jsonl_files[0].read_text("utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            transcript.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                return transcript
        return []

    def _transcript_to_messages(self, transcript: list[dict]) -> list[dict]:
        msgs = []
        for entry in transcript:
            if entry.get("type") != "message":
                continue
            msg = entry.get("message", {})
            role = msg.get("role")
            if role in ("user", "assistant", "tool", "system"):
                msgs.append({
                    "role": role, "content": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls", []),
                })
        return msgs

    def _run_grading(self, task_id: str, transcript: list[dict]) -> bool:
        if not self.oc_config.pinchbench_dir:
            return False
        try:
            import sys, subprocess
            scripts_dir = Path(self.oc_config.pinchbench_dir) / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from lib_tasks import TaskLoader
            from lib_grading import grade_task
            tasks_dir = Path(self.oc_config.pinchbench_dir) / "tasks"
            loader = TaskLoader(tasks_dir)
            task_file = tasks_dir / f"{task_id}.md"
            if not task_file.exists():
                return False
            task = loader.load_task(task_file)
            if task is None:
                return False

            workspace = Path(self.oc_config.workspace_base) / task_id
            workspace.mkdir(parents=True, exist_ok=True)

            # Sync workspace files from ECS (files are created there by OpenClaw)
            if self.oc_config.host not in ("localhost", "127.0.0.1"):
                ssh_key = self.oc_config.ssh_key or "/root/.ssh/id_ed25519"
                remote_ws = f"{self.oc_config.workspace_base}/{task_id}/"
                try:
                    subprocess.run(
                        ["rsync", "-az", "--timeout=10",
                         "-e", f"ssh -o StrictHostKeyChecking=no -i {ssh_key}",
                         f"{self.oc_config.user}@{self.oc_config.host}:{remote_ws}",
                         str(workspace) + "/"],
                        capture_output=True, timeout=30,
                    )
                    logger.info("Synced workspace from ECS for %s", task_id)
                except Exception as e:
                    logger.warning("rsync workspace failed for %s: %s", task_id, e)

            execution_result = {
                "transcript": transcript,
                "workspace": str(workspace),
                "status": "completed",
            }
            skill_dir = Path(self.oc_config.pinchbench_dir)
            judge_model = os.environ.get("PINCHBENCH_GRADE_JUDGE_MODEL", "qwen-plus")
            judge_backend = os.environ.get("PINCHBENCH_GRADE_JUDGE_BACKEND", "api")
            judge_base_url = os.environ.get(
                "PINCHBENCH_GRADE_JUDGE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            judge_api_key = os.environ.get(
                "PINCHBENCH_GRADE_JUDGE_API_KEY",
                os.environ.get("JUDGE_API_KEY", os.environ.get("DASHSCOPE_API_KEY", "")),
            )
            if judge_backend == "api" and judge_api_key:
                result = grade_task(
                    task=task,
                    execution_result=execution_result,
                    skill_dir=skill_dir,
                    judge_model=judge_model,
                    judge_backend=judge_backend,
                    judge_base_url=judge_base_url,
                    judge_api_key=judge_api_key,
                )
            else:
                result = grade_task(task=task, execution_result=execution_result, skill_dir=skill_dir)
            score = getattr(result, "score", 0.0)
            logger.info("Grading %s: score=%.2f", task_id, score)
            return score >= 0.5
        except Exception as e:
            logger.error("Grading failed for %s: %s", task_id, e)
            return False

    # ── Reward computation ──

    def _get_vllm_base_url(self) -> str:
        """Get actual vLLM HTTP URL from server_manager (Ray-assigned dynamic port)."""
        try:
            addrs = list(self.server_manager._server_id_to_handle.keys())
            if addrs:
                url = f"http://{addrs[0]}/v1"
                logger.info("Detected vLLM URL: %s", url)
                return url
        except Exception:
            pass
        return self.oc_config.prm_vllm_base_url

    async def _compute_rewards(
        self, trajectory: list[dict], terminal_success: bool, task_id: str, task_prompt: str,
    ) -> list[float]:
        try:
            from .reward import compute_episode_rewards_async
            vllm_url = self._get_vllm_base_url()
            return await compute_episode_rewards_async(
                trajectory, terminal_success, task_id,
                task_prompt=task_prompt,
                mode=self.oc_config.reward_mode,
                vllm_base_url=vllm_url,
                judge_model=self.oc_config.prm_model,
                judge_api_key=self.oc_config.prm_api_key,
            )
        except Exception as e:
            logger.error("Reward computation failed: %s", e)
            return []

    def _assign_rewards(
        self, response_ids: list[int], response_mask: list[int],
        per_turn_rewards: list[float], terminal_reward: float,
    ) -> list[float]:
        rewards = [0.0] * len(response_ids)
        if not response_ids:
            return rewards

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        model_im_ends = [
            i for i, tid in enumerate(response_ids)
            if tid == im_end_id and i < len(response_mask) and response_mask[i] == 1
        ]

        n = min(len(per_turn_rewards), len(model_im_ends))
        for k in range(n):
            rewards[model_im_ends[k]] = per_turn_rewards[k]

        if n < len(per_turn_rewards):
            leftover = sum(per_turn_rewards[n:])
            rewards[-1] += leftover

        rewards[-1] += terminal_reward
        return rewards

    # ── Utilities ──

    def _parse_tool_calls(self, text: str) -> Optional[list[dict]]:
        import re
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

    def _prepare_messages(self, messages: list[dict], tools: list[dict] | None) -> list[dict]:
        """Prepare OpenClaw messages for Qwen's apply_chat_template.

        1. Flatten content-parts format (list of dicts) to plain strings,
           since Qwen's template silently drops list-format content.
        2. Inject <tool_call> format instructions so the model knows to output
           parseable XML tags when calling tools.
        """
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

        if tools:
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

    def _strip_tool_tags(self, text: str) -> str:
        """Strip <tool_call>...</tool_call> and <think>...</think> from text for OpenClaw content field."""
        import re
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _get_public_ip(self) -> str:
        """Get public IP (kept for fallback/debugging)."""
        try:
            import urllib.request
            return urllib.request.urlopen("https://ifconfig.me", timeout=5).read().decode().strip()
        except Exception:
            return "127.0.0.1"
