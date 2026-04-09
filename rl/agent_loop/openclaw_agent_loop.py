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
    agent_timeout: float = 600.0
    max_turns: int = 30
    prm_vllm_base_url: str = "http://localhost:8000/v1"
    prm_model: str = "Qwen3-4B"
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
            agent_timeout=float(os.environ.get("AGENT_TIMEOUT", "600")),
            max_turns=int(os.environ.get("MAX_TURNS", "30")),
            prm_vllm_base_url=os.environ.get("PRM_VLLM_BASE_URL", "http://localhost:8000/v1"),
            prm_model=os.environ.get("PRM_MODEL", "Qwen3-4B"),
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

        raw_prompt = kwargs.get("raw_prompt", [])
        extra_info = kwargs.get("extra_info", {})
        task_id = extra_info.get("task_id", "unknown")
        task_prompt = ""
        if raw_prompt:
            last_user = [m for m in raw_prompt if m.get("role") == "user"]
            if last_user:
                task_prompt = last_user[-1].get("content", "")

        session_id = f"rl-{uuid.uuid4().hex[:8]}"

        proxy = ModelProxy(host=self.oc_config.proxy_bind_host, port=0)
        proxy_port = await proxy.start()

        all_prompt_ids: list[int] = []
        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        all_response_logprobs: list[float] = []
        turns: list[TurnRecord] = []
        messages: list[dict] = []
        turn_count = 0

        t_start = time.time()
        openclaw_proc: Optional[asyncio.subprocess.Process] = None
        metrics: dict[str, Any] = {}

        try:
            if self.oc_config.host in ("localhost", "127.0.0.1"):
                openclaw_proc = await self._start_local_openclaw(
                    task_prompt, session_id, f"http://127.0.0.1:{proxy_port}/v1", task_id,
                )
            else:
                openclaw_proc = await self._start_remote_openclaw(
                    task_prompt, session_id, proxy_port, task_id,
                )

            while turn_count < self.oc_config.max_turns:
                try:
                    req: ModelRequest = await asyncio.wait_for(
                        proxy.get_request(), timeout=self.oc_config.agent_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Proxy timeout at turn %d", turn_count)
                    break

                if openclaw_proc.returncode is not None:
                    logger.info("OpenClaw exited (code=%s)", openclaw_proc.returncode)
                    req.response_error = "agent process exited"
                    await proxy.send_response(req)
                    break

                # Apply chat template to tokenize
                chat_messages = list(req.messages)
                try:
                    prompt_token_ids = await self.apply_chat_template(chat_messages)
                except Exception as e:
                    logger.error("Chat template failed: %s", e)
                    req.response_error = str(e)
                    await proxy.send_response(req)
                    break

                if turn_count == 0:
                    all_prompt_ids = list(prompt_token_ids)

                # Generate via veRL's server_manager
                gen_output = await self.server_manager.generate(
                    request_id=uuid.uuid4().hex,
                    prompt_ids=prompt_token_ids,
                    sampling_params=sampling_params,
                )

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

                req.response_text = clean_text
                req.response_tool_calls = tool_calls
                req.finish_reason = "tool_calls" if tool_calls else "stop"
                await proxy.send_response(req)

                messages.append({"role": "assistant", "content": clean_text, "tool_calls": tool_calls or []})
                turn_count += 1

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
        total_reward = sum(per_turn_rewards) + terminal_reward

        # Assign per-turn rewards at <|im_end|> token positions
        reward_at_tokens = self._assign_rewards(
            all_response_ids, all_response_mask, per_turn_rewards, terminal_reward,
        )

        response_length = self.rollout_config.response_length
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
             "--model", "verl-proxy", "--workspace", str(workspace), "--non-interactive"],
            capture_output=True, text=True, check=False,
        )
        agent_dir = Path.home() / ".openclaw" / "agents" / agent_id / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        models = {
            "providers": {"verl": {
                "baseUrl": proxy_url, "apiKey": "dummy", "api": "openai-completions",
                "models": [{"id": "verl-proxy", "name": "verl-proxy"}],
            }},
            "defaultProvider": "verl", "defaultModel": "verl-proxy",
        }
        (agent_dir / "models.json").write_text(json.dumps(models, indent=2), "utf-8")
        sessions_file = Path.home() / ".openclaw" / "agents" / agent_id / "sessions" / "sessions.json"
        if sessions_file.exists():
            sessions_file.unlink()

    def _build_remote_setup(self, agent_id: str, proxy_url: str, workspace: str, task_id: str) -> str:
        models_json = json.dumps({
            "providers": {"verl": {
                "baseUrl": proxy_url, "apiKey": "dummy", "api": "openai-completions",
                "models": [{"id": "verl-proxy", "name": "verl-proxy"}],
            }},
            "defaultProvider": "verl", "defaultModel": "verl-proxy",
        })
        return " && ".join([
            f"mkdir -p {workspace}",
            f"openclaw agents add {agent_id} --model verl-proxy --workspace {workspace} --non-interactive 2>/dev/null || true",
            f"mkdir -p ~/.openclaw/agents/{agent_id}/agent",
            f"echo '{models_json}' > ~/.openclaw/agents/{agent_id}/agent/models.json",
            f"rm -f ~/.openclaw/agents/{agent_id}/sessions/sessions.json",
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
            loader = TaskLoader(Path(self.oc_config.pinchbench_dir) / "tasks")
            task = loader.get_task(task_id)
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
            import sys
            scripts_dir = Path(self.oc_config.pinchbench_dir) / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from lib_tasks import TaskLoader
            from lib_grading import grade_task
            loader = TaskLoader(Path(self.oc_config.pinchbench_dir) / "tasks")
            task = loader.get_task(task_id)
            if task is None:
                return False
            workspace = Path(self.oc_config.workspace_base) / task_id
            result = grade_task(task=task, transcript=transcript, workspace=workspace)
            score = result.get("total_score", 0.0)
            logger.info("Grading %s: score=%.2f", task_id, score)
            return score >= 0.5
        except Exception as e:
            logger.error("Grading failed for %s: %s", task_id, e)
            return False

    # ── Reward computation ──

    async def _compute_rewards(
        self, trajectory: list[dict], terminal_success: bool, task_id: str, task_prompt: str,
    ) -> list[float]:
        try:
            from .reward import compute_episode_rewards_async
            return await compute_episode_rewards_async(
                trajectory, terminal_success, task_id,
                task_prompt=task_prompt,
                mode=self.oc_config.reward_mode,
                vllm_base_url=self.oc_config.prm_vllm_base_url,
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

    def _get_public_ip(self) -> str:
        """Get public IP (kept for fallback/debugging)."""
        try:
            import urllib.request
            return urllib.request.urlopen("https://ifconfig.me", timeout=5).read().decode().strip()
        except Exception:
            return "127.0.0.1"
