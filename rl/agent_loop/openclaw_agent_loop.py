"""
OpenClawAgentLoop: veRL agent loop that drives OpenClaw as the RL environment.

Architecture (SWE-Agent recipe pattern):
  1. Start ModelProxy on ephemeral port
  2. SSH to remote host (ECS/Mac), run `openclaw agent --local`
     with models.json pointing baseUrl at ModelProxy
  3. Intercept each LLM request from OpenClaw via ModelProxy
  4. Forward to veRL's vLLM inference engine for generation
  5. Return response to OpenClaw, which executes tools locally
  6. After episode ends, run PinchBench grading for terminal reward
  7. Compute per-turn process rewards from rubrics
  8. Reconstruct token-level trajectory for veRL training

Environment variables:
  OPENCLAW_HOST       - SSH host for OpenClaw (default: localhost)
  OPENCLAW_USER       - SSH user (default: root)
  OPENCLAW_SSH_KEY    - SSH key path (default: ~/.ssh/id_ed25519)
  OPENCLAW_PORT       - SSH port (default: 22)
  OPENCLAW_WORKSPACE  - workspace base dir on remote (default: /tmp/pinchbench)
  PINCHBENCH_DIR      - local pinchbench-skill repo root
  JUDGE_MODEL         - judge model for llm_judge grading (default: qwen-plus)
  JUDGE_BASE_URL      - judge API base URL
  JUDGE_API_KEY       - judge API key
  REWARD_MODE         - "baseline", "rule", or "oracle" (default: oracle)
  PROXY_BIND_HOST     - ModelProxy bind address (default: 0.0.0.0)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class OpenClawConfig:
    host: str = "localhost"
    user: str = "root"
    ssh_key: str = str(Path.home() / ".ssh" / "id_ed25519")
    ssh_port: int = 22
    workspace_base: str = "/tmp/pinchbench"
    pinchbench_dir: str = ""
    judge_model: str = "qwen-plus"
    judge_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    judge_api_key: str = ""
    reward_mode: str = "self-judge"
    proxy_bind_host: str = "0.0.0.0"
    agent_timeout: float = 600.0
    max_turns: int = 30
    # PRM self-judge: Qwen3-4B scores its own turns via vLLM
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
            judge_model=os.environ.get("JUDGE_MODEL", "qwen-plus"),
            judge_base_url=os.environ.get(
                "JUDGE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            judge_api_key=os.environ.get("JUDGE_API_KEY", ""),
            reward_mode=os.environ.get("REWARD_MODE", "self-judge"),
            proxy_bind_host=os.environ.get("PROXY_BIND_HOST", "0.0.0.0"),
            agent_timeout=float(os.environ.get("AGENT_TIMEOUT", "600")),
            max_turns=int(os.environ.get("MAX_TURNS", "30")),
            prm_vllm_base_url=os.environ.get("PRM_VLLM_BASE_URL", "http://localhost:8000/v1"),
            prm_model=os.environ.get("PRM_MODEL", "Qwen3-4B"),
            prm_api_key=os.environ.get("PRM_API_KEY", "dummy"),
        )


class OpenClawAgentLoop:
    """Drives OpenClaw as an RL environment for veRL training.

    This class can be used standalone (without veRL's AgentLoopBase)
    for testing, or integrated into veRL via a thin wrapper that
    inherits AgentLoopBase.

    Usage:
        loop = OpenClawAgentLoop(config, tokenizer, generate_fn)
        output = await loop.run(task_id, task_prompt, sampling_params)
    """

    def __init__(
        self,
        config: OpenClawConfig,
        tokenizer: Any,
        generate_fn: Any = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.generate_fn = generate_fn

        from .model_proxy import ModelProxy
        from .trajectory import TrajectoryReconstructor

        self._proxy_cls = ModelProxy
        self._reconstructor = TrajectoryReconstructor(tokenizer)

    async def run(
        self,
        task_id: str,
        task_prompt: str,
        sampling_params: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run a single episode: OpenClaw executes one PinchBench task.

        Returns dict with:
          - prompt_ids, response_ids, response_mask, response_logprobs
          - per_turn_rewards, terminal_reward, total_reward
          - num_turns, task_id, trajectory
        """
        from .model_proxy import ModelProxy, ModelRequest
        from .trajectory import TurnRecord
        from .reward import compute_episode_rewards_async

        session_id = session_id or f"rl-{uuid.uuid4().hex[:8]}"
        proxy = ModelProxy(host=self.config.proxy_bind_host, port=0)
        proxy_port = await proxy.start()

        turns: list[TurnRecord] = []
        transcript_raw: list[dict] = []
        episode_done = asyncio.Event()
        openclaw_proc: Optional[asyncio.subprocess.Process] = None

        try:
            # Determine proxy URL that OpenClaw can reach
            if self.config.host == "localhost" or self.config.host == "127.0.0.1":
                proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
                openclaw_proc = await self._start_local_openclaw(
                    task_prompt, session_id, proxy_url, task_id,
                )
            else:
                proxy_url = f"http://{self.config.proxy_bind_host}:{proxy_port}/v1"
                openclaw_proc = await self._start_remote_openclaw(
                    task_prompt, session_id, proxy_url, task_id, proxy_port,
                )

            # Main interaction loop
            turn_count = 0
            while turn_count < self.config.max_turns:
                try:
                    req: ModelRequest = await asyncio.wait_for(
                        proxy.get_request(), timeout=self.config.agent_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("ModelProxy timeout waiting for request (turn %d)", turn_count)
                    break

                # Check if OpenClaw process has exited
                if openclaw_proc.returncode is not None:
                    logger.info("OpenClaw process exited (code=%s)", openclaw_proc.returncode)
                    req.response_error = "agent process exited"
                    await proxy.send_response(req)
                    break

                # Tokenize the messages
                messages = req.messages
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                except Exception as e:
                    logger.error("Tokenization failed: %s", e)
                    req.response_error = str(e)
                    await proxy.send_response(req)
                    break

                # Generate with veRL's vLLM or external generate function
                if self.generate_fn:
                    gen_output = await self.generate_fn(
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params or {},
                    )
                    response_ids = gen_output.get("token_ids", [])
                    response_logprobs = gen_output.get("logprobs", [])
                else:
                    # Fallback: no generation function (testing mode)
                    logger.warning("No generate_fn provided, returning empty response")
                    req.response_error = "no generate function"
                    await proxy.send_response(req)
                    break

                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                # Parse tool calls from generated text if present
                tool_calls = self._parse_tool_calls(response_text)

                turns.append(TurnRecord(
                    turn_index=turn_count,
                    messages=messages,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_text=response_text,
                    response_logprobs=response_logprobs if response_logprobs else None,
                ))

                # Send response back to OpenClaw via proxy
                req.response_text = response_text
                req.response_tool_calls = tool_calls
                req.finish_reason = "tool_calls" if tool_calls else "stop"
                await proxy.send_response(req)

                turn_count += 1

            # Wait for OpenClaw process to finish
            if openclaw_proc and openclaw_proc.returncode is None:
                try:
                    await asyncio.wait_for(openclaw_proc.wait(), timeout=30)
                except asyncio.TimeoutError:
                    openclaw_proc.kill()
                    await openclaw_proc.wait()

        finally:
            await proxy.drain()
            await proxy.stop()

        # Load transcript from OpenClaw sessions
        transcript_raw = self._load_openclaw_transcript(session_id, task_id)

        # Run PinchBench grading
        terminal_success = self._run_grading(task_id, transcript_raw)

        # Compute per-turn rewards (self-judge: Qwen3-4B scores its own turns)
        trajectory_for_reward = self._transcript_to_messages(transcript_raw)
        per_turn_rewards = await compute_episode_rewards_async(
            trajectory_for_reward,
            terminal_success,
            task_id,
            task_prompt=task_prompt,
            mode=self.config.reward_mode,
            vllm_base_url=self.config.prm_vllm_base_url,
            judge_model=self.config.prm_model,
            judge_api_key=self.config.prm_api_key,
        )

        # Reconstruct token-level trajectory
        aligned = self._reconstructor.reconstruct(turns)

        # Assign per-turn rewards to token positions
        reward_at_tokens = self._assign_rewards_to_tokens(
            aligned, per_turn_rewards
        )

        terminal_reward = 1.0 if terminal_success else -1.0

        return {
            "prompt_ids": aligned.initial_prompt_ids,
            "response_ids": aligned.response_ids,
            "response_mask": aligned.response_mask,
            "response_logprobs": aligned.response_logprobs,
            "reward_at_tokens": reward_at_tokens,
            "per_turn_rewards": per_turn_rewards,
            "terminal_reward": terminal_reward,
            "terminal_success": terminal_success,
            "total_reward": sum(per_turn_rewards),
            "num_turns": aligned.num_turns,
            "task_id": task_id,
            "trajectory": trajectory_for_reward,
            "alignment_ok": aligned.ok,
        }

    # ── OpenClaw process management ──

    async def _start_local_openclaw(
        self,
        prompt: str,
        session_id: str,
        proxy_url: str,
        task_id: str,
    ) -> asyncio.subprocess.Process:
        """Start OpenClaw locally (same machine as trainer)."""
        agent_id = f"rl-{task_id}-{session_id[:8]}"
        workspace = Path(self.config.workspace_base) / task_id

        self._setup_openclaw_agent(agent_id, proxy_url, workspace)
        self._prepare_workspace(task_id, workspace)

        proc = await asyncio.create_subprocess_exec(
            "openclaw", "agent",
            "--agent", agent_id,
            "--session-id", session_id,
            "--message", prompt,
            "--local",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )
        logger.info(
            "Started local OpenClaw (pid=%s, agent=%s, task=%s)",
            proc.pid, agent_id, task_id,
        )
        return proc

    async def _start_remote_openclaw(
        self,
        prompt: str,
        session_id: str,
        proxy_url: str,
        task_id: str,
        proxy_port: int,
    ) -> asyncio.subprocess.Process:
        """Start OpenClaw on remote host via SSH."""
        agent_id = f"rl-{task_id}-{session_id[:8]}"
        workspace = f"{self.config.workspace_base}/{task_id}"

        # Build remote setup + execution command
        # The proxy_url uses the RunPod's public IP so ECS can reach it
        runpod_ip = self._get_public_ip()
        remote_proxy_url = f"http://{runpod_ip}:{proxy_port}/v1"

        setup_cmd = self._build_remote_setup_cmd(
            agent_id, remote_proxy_url, workspace, task_id,
        )
        run_cmd = (
            f"openclaw agent "
            f"--agent {agent_id} "
            f"--session-id {session_id} "
            f'--message "{self._escape_for_ssh(prompt)}" '
            f"--local"
        )

        full_cmd = f"{setup_cmd} && cd {workspace} && {run_cmd}"

        proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-i", self.config.ssh_key,
            "-p", str(self.config.ssh_port),
            f"{self.config.user}@{self.config.host}",
            full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info(
            "Started remote OpenClaw on %s (pid=%s, agent=%s, task=%s)",
            self.config.host, proc.pid, agent_id, task_id,
        )
        return proc

    def _build_remote_setup_cmd(
        self,
        agent_id: str,
        proxy_url: str,
        workspace: str,
        task_id: str,
    ) -> str:
        """Build shell commands to set up OpenClaw agent on remote host."""
        models_json = json.dumps({
            "providers": {
                "verl": {
                    "baseUrl": proxy_url,
                    "apiKey": "dummy",
                    "api": "openai-completions",
                    "models": [{"id": "verl-proxy", "name": "verl-proxy"}],
                }
            },
            "defaultProvider": "verl",
            "defaultModel": "verl-proxy",
        })

        commands = [
            f"mkdir -p {workspace}",
            f"openclaw agents add {agent_id} --model verl-proxy --workspace {workspace} --non-interactive 2>/dev/null || true",
            f"mkdir -p ~/.openclaw/agents/{agent_id}/agent",
            f"echo '{models_json}' > ~/.openclaw/agents/{agent_id}/agent/models.json",
            f"rm -f ~/.openclaw/agents/{agent_id}/sessions/sessions.json",
        ]

        # Copy task fixtures if pinchbench_dir is configured
        if self.config.pinchbench_dir:
            commands.append(
                f"cd {self.config.pinchbench_dir} && "
                f"python3 -c \"import sys; sys.path.insert(0,'scripts'); "
                f"from lib_tasks import TaskLoader; "
                f"t = TaskLoader('{self.config.pinchbench_dir}/tasks').get_task('{task_id}'); "
                f"print('ok')\" 2>/dev/null || true"
            )

        return " && ".join(commands)

    def _setup_openclaw_agent(
        self, agent_id: str, proxy_url: str, workspace: Path
    ) -> None:
        """Set up OpenClaw agent locally with proxy URL."""
        workspace.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["openclaw", "agents", "add", agent_id,
             "--model", "verl-proxy",
             "--workspace", str(workspace),
             "--non-interactive"],
            capture_output=True, text=True, check=False,
        )

        agent_dir = Path.home() / ".openclaw" / "agents" / agent_id / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)

        models_json = {
            "providers": {
                "verl": {
                    "baseUrl": proxy_url,
                    "apiKey": "dummy",
                    "api": "openai-completions",
                    "models": [{"id": "verl-proxy", "name": "verl-proxy"}],
                }
            },
            "defaultProvider": "verl",
            "defaultModel": "verl-proxy",
        }
        (agent_dir / "models.json").write_text(
            json.dumps(models_json, indent=2), "utf-8"
        )

        # Clear stale sessions
        sessions_file = (
            Path.home() / ".openclaw" / "agents" / agent_id / "sessions" / "sessions.json"
        )
        if sessions_file.exists():
            sessions_file.unlink()

    def _prepare_workspace(self, task_id: str, workspace: Path) -> None:
        """Prepare task workspace with fixture files."""
        if not self.config.pinchbench_dir:
            return

        import sys
        scripts_dir = Path(self.config.pinchbench_dir) / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from lib_tasks import TaskLoader
            loader = TaskLoader(Path(self.config.pinchbench_dir) / "tasks")
            task = loader.get_task(task_id)
            if task is None:
                logger.warning("Task %s not found in PinchBench", task_id)
                return

            import shutil
            if workspace.exists():
                shutil.rmtree(workspace)
            workspace.mkdir(parents=True, exist_ok=True)

            for file_spec in task.workspace_files:
                if "content" in file_spec:
                    dest = workspace / file_spec["path"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(file_spec["content"])
                elif "source" in file_spec:
                    src = Path(self.config.pinchbench_dir) / "assets" / file_spec["source"]
                    dest = workspace / file_spec["dest"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(src.read_bytes())

        except ImportError:
            logger.warning("Could not import lib_tasks, skipping workspace setup")
        except Exception as e:
            logger.error("Workspace preparation failed: %s", e)

    # ── Transcript loading ──

    def _load_openclaw_transcript(
        self, session_id: str, task_id: str
    ) -> list[dict]:
        """Load transcript from OpenClaw sessions directory."""
        agents_dir = Path.home() / ".openclaw" / "agents"
        agent_id_prefix = f"rl-{task_id}"

        for agent_dir in agents_dir.iterdir():
            if not agent_dir.name.startswith(agent_id_prefix):
                continue
            sessions_dir = agent_dir / "sessions"
            if not sessions_dir.exists():
                continue

            jsonl_files = sorted(
                sessions_dir.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if jsonl_files:
                transcript = []
                for line in jsonl_files[0].read_text("utf-8").splitlines():
                    if line.strip():
                        try:
                            transcript.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                return transcript

        logger.warning("No transcript found for session %s", session_id)
        return []

    def _transcript_to_messages(self, transcript: list[dict]) -> list[dict]:
        """Convert OpenClaw JSONL transcript to flat message list."""
        messages = []
        for entry in transcript:
            if entry.get("type") != "message":
                continue
            msg = entry.get("message", {})
            role = msg.get("role")
            if role in ("user", "assistant", "tool", "system"):
                messages.append({
                    "role": role,
                    "content": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls", []),
                    "tool_name": msg.get("name"),
                })
        return messages

    # ── Grading ──

    def _run_grading(self, task_id: str, transcript: list[dict]) -> bool:
        """Run PinchBench grading to get terminal reward."""
        if not self.config.pinchbench_dir:
            logger.warning("No pinchbench_dir configured, skipping grading")
            return False

        try:
            import sys
            scripts_dir = Path(self.config.pinchbench_dir) / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            from lib_tasks import TaskLoader
            from lib_grading import grade_task

            loader = TaskLoader(Path(self.config.pinchbench_dir) / "tasks")
            task = loader.get_task(task_id)
            if task is None:
                return False

            workspace = Path(self.config.workspace_base) / task_id
            result = grade_task(
                task=task,
                transcript=transcript,
                workspace=workspace,
                judge_model=self.config.judge_model,
                judge_base_url=self.config.judge_base_url,
                judge_api_key=self.config.judge_api_key,
            )
            score = result.get("total_score", 0.0)
            logger.info("Grading %s: score=%.2f", task_id, score)
            return score >= 0.5

        except Exception as e:
            logger.error("Grading failed for %s: %s", task_id, e)
            return False

    # ── Reward assignment ──

    def _assign_rewards_to_tokens(
        self,
        aligned: Any,
        per_turn_rewards: list[float],
    ) -> list[float]:
        """Assign per-turn rewards to token positions (at <|im_end|>)."""
        reward_at_tokens = [0.0] * len(aligned.response_ids)

        if not aligned.ok:
            if reward_at_tokens:
                reward_at_tokens[-1] = sum(per_turn_rewards) if per_turn_rewards else -1.0
            return reward_at_tokens

        # Find <|im_end|> positions in model-generated tokens
        im_end_positions = self._reconstructor.find_assistant_turn_ends(
            aligned.response_ids
        )

        # Only count positions where mask=1 (model-generated)
        model_im_ends = [
            pos for pos in im_end_positions
            if pos < len(aligned.response_mask) and aligned.response_mask[pos] == 1
        ]

        n = min(len(per_turn_rewards), len(model_im_ends))
        for k in range(n):
            reward_at_tokens[model_im_ends[k]] = per_turn_rewards[k]

        # Leftover rewards go to last token
        if n < len(per_turn_rewards):
            leftover = sum(per_turn_rewards[n:])
            if reward_at_tokens:
                reward_at_tokens[-1] += leftover

        return reward_at_tokens

    # ── Utilities ──

    def _parse_tool_calls(self, text: str) -> Optional[list[dict]]:
        """Parse tool calls from Hermes-format generated text."""
        import re

        tool_calls = []
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                tc = json.loads(match.group(1))
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(
                            tc.get("arguments", {}), ensure_ascii=False
                        ),
                    },
                })
            except json.JSONDecodeError:
                pass

        return tool_calls if tool_calls else None

    def _escape_for_ssh(self, text: str) -> str:
        """Escape text for safe SSH command execution."""
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")

    def _get_public_ip(self) -> str:
        """Get public IP of this machine (for remote OpenClaw to reach proxy)."""
        try:
            import urllib.request
            return urllib.request.urlopen(
                "https://ifconfig.me", timeout=5
            ).read().decode().strip()
        except Exception:
            return "127.0.0.1"
