# Loaded automatically by Python if this repo root is on sys.path (see PYTHONPATH in run_reinforce_lora.sh).
# Ray workers inherit env + PYTHONPATH, so the patch applies inside TaskRunner too.

from __future__ import annotations

import os

def _maybe_patch_verl_best_ckpt() -> None:
    v = os.environ.get("PINCHBENCH_BEST_CKPT", "").strip().lower()
    if v not in ("1", "true", "yes", "on"):
        return
    try:
        from rl.verl_best_ckpt_patch import apply_patch

        apply_patch()
    except Exception as e:
        import sys

        print(f"[sitecustomize] PINCHBENCH_BEST_CKPT patch skipped: {e}", file=sys.stderr)


_maybe_patch_verl_best_ckpt()
