#!/usr/bin/env python3
"""Force Ray dashboard agent to start in minimal mode.

Ray 2.55 on our A100 pod can fail during raylet startup with:

  Timed out waiting for file .../metrics_agent_port_<node_id>

The dashboard agent writes this file very early in `--minimal` mode, but much
later in the full mode after loading many modules. This patch appends
`--minimal` to the generated dashboard agent command in the installed Ray
services module so raylet can observe the port file before its internal timeout.
"""

from __future__ import annotations

from pathlib import Path
import sys


TARGET = Path("/usr/local/lib/python3.12/dist-packages/ray/_private/services.py")
BACKUP = TARGET.with_suffix(TARGET.suffix + ".pinchbench.bak")

NEEDLE = """    if ray._private.utils.get_dashboard_dependency_error() is not None:
        # If dependencies are not installed, it is the minimally packaged
        # ray. We should restrict the features within dashboard agent
        # that requires additional dependencies to be downloaded.
        dashboard_agent_command.append("--minimal")
"""

REPLACEMENT = """    if ray._private.utils.get_dashboard_dependency_error() is not None:
        # If dependencies are not installed, it is the minimally packaged
        # ray. We should restrict the features within dashboard agent
        # that requires additional dependencies to be downloaded.
        dashboard_agent_command.append("--minimal")
    elif "--minimal" not in dashboard_agent_command:
        dashboard_agent_command.append("--minimal")
"""


def main() -> int:
    text = TARGET.read_text()
    if REPLACEMENT in text:
        print(f"[patch_ray_minimal_dashboard_agent] already patched: {TARGET}")
        return 0
    if NEEDLE not in text:
        print(f"[patch_ray_minimal_dashboard_agent] needle not found: {TARGET}", file=sys.stderr)
        return 1
    if not BACKUP.exists():
        BACKUP.write_text(text)
    TARGET.write_text(text.replace(NEEDLE, REPLACEMENT, 1))
    print(f"[patch_ray_minimal_dashboard_agent] patched: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
