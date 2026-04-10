#!/usr/bin/env python3
"""Seed ECS /tmp/pinchbench/<task_id>/ from task YAML workspace_files (same as training).

Used by E2E tests so OpenClaw on ECS sees inbox files etc.

Env: PINCHBENCH_DIR, OPENCLAW_HOST, OPENCLAW_USER, OPENCLAW_SSH_KEY, OPENCLAW_PORT,
     OPENCLAW_WORKSPACE (default /tmp/pinchbench)
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: seed_ecs_workspace.py <task_id>", file=sys.stderr)
        sys.exit(2)
    task_id = sys.argv[1]
    pb = os.environ.get("PINCHBENCH_DIR", "").strip()
    if not pb:
        print("PINCHBENCH_DIR is required", file=sys.stderr)
        sys.exit(1)
    host = os.environ.get("OPENCLAW_HOST", "").strip()
    if not host or host in ("localhost", "127.0.0.1"):
        print("OPENCLAW_HOST must be set to ECS IP", file=sys.stderr)
        sys.exit(1)
    user = os.environ.get("OPENCLAW_USER", "root")
    port = os.environ.get("OPENCLAW_PORT", "22")
    key = os.environ.get("OPENCLAW_SSH_KEY", str(Path.home() / ".ssh/id_ed25519"))
    base = os.environ.get("OPENCLAW_WORKSPACE", "/tmp/pinchbench")
    workspace = f"{base}/{task_id}"

    scripts_dir = Path(pb) / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from lib_tasks import TaskLoader, resolve_task_markdown_path  # noqa: E402

    tasks_dir = Path(pb) / "tasks"
    task_file = resolve_task_markdown_path(tasks_dir, task_id)
    loader = TaskLoader(tasks_dir)
    task = loader.load_task(task_file)
    if task is None:
        print(f"Task not found: {task_id}", file=sys.stderr)
        sys.exit(1)
    files = getattr(task, "workspace_files", []) or []
    if not files:
        print(f"[seed] no workspace_files for {task_id}, skipping rsync")
        return

    with tempfile.TemporaryDirectory(prefix="pinchbench_seed_") as td:
        local_root = Path(td) / task_id
        local_root.mkdir(parents=True)
        for spec in files:
            if "content" in spec:
                dest = local_root / spec["path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(spec["content"])
            elif "source" in spec:
                src = Path(pb) / "assets" / spec["source"]
                dest = local_root / spec.get("dest", spec["source"])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())

        ws = shlex.quote(workspace)
        r = subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=10",
                "-i",
                key,
                "-p",
                str(port),
                f"{user}@{host}",
                f"rm -rf {ws} && mkdir -p {ws}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            print(r.stderr or r.stdout, file=sys.stderr)
            sys.exit(r.returncode)
        rsync = [
            "rsync",
            "-az",
            "--timeout=60",
            "-e",
            f"ssh -o StrictHostKeyChecking=no -i {key} -p {port}",
            f"{local_root}/",
            f"{user}@{host}:{workspace}/",
        ]
        r2 = subprocess.run(rsync, capture_output=True, text=True, timeout=120)
        if r2.returncode != 0:
            print(r2.stderr or r2.stdout, file=sys.stderr)
            sys.exit(r2.returncode)
        print(f"[seed] OK {task_id} -> {host}:{workspace}")


if __name__ == "__main__":
    main()
