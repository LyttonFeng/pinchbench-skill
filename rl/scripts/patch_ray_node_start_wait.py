#!/usr/bin/env python3
"""Patch installed Ray to make node startup wait configurable.

Ray 2.55 hardcodes:

    raylet_start_wait_time_s = 30

inside `ray/_private/node.py`, which is too short for our container. This patch
switches it to:

    raylet_start_wait_time_s = int(os.environ.get("RAY_raylet_start_wait_time_s", "180"))
"""

from __future__ import annotations

from pathlib import Path
import sys


TARGET = Path("/usr/local/lib/python3.12/dist-packages/ray/_private/node.py")
BACKUP = TARGET.with_suffix(TARGET.suffix + ".pinchbench.bak")

NEEDLE = """            raylet_start_wait_time_s = 30
"""

REPLACEMENT = """            raylet_start_wait_time_s = int(
                os.environ.get("RAY_raylet_start_wait_time_s", "180")
            )
"""


def main() -> int:
    text = TARGET.read_text()
    if REPLACEMENT in text:
        print(f"[patch_ray_node_start_wait] already patched: {TARGET}")
        return 0
    if NEEDLE not in text:
        print(f"[patch_ray_node_start_wait] needle not found: {TARGET}", file=sys.stderr)
        return 1
    if not BACKUP.exists():
        BACKUP.write_text(text)
    TARGET.write_text(text.replace(NEEDLE, REPLACEMENT, 1))
    print(f"[patch_ray_node_start_wait] patched: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
