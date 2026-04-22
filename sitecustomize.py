# Loaded automatically by Python if this repo root is on sys.path (see PYTHONPATH in run_reinforce_lora.sh).
# Ray workers inherit env + PYTHONPATH, so the patch applies inside TaskRunner too.

from __future__ import annotations

import os
from pathlib import Path


def _ensure_pinchbench_dir_env() -> None:
    """Ray workers sometimes miss shell exports; infer repo root so ECS workspace seeding finds tasks/assets."""
    if os.environ.get("PINCHBENCH_DIR", "").strip():
        return
    root = Path(__file__).resolve().parent
    if (root / "tasks").is_dir() and (root / "assets").is_dir():
        os.environ["PINCHBENCH_DIR"] = str(root)


_ensure_pinchbench_dir_env()

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


def _maybe_patch_verl_debug_metrics() -> None:
    v = os.environ.get("PINCHBENCH_DEBUG_METRICS_PATCH", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return
    try:
        from rl.verl_debug_metrics_patch import apply_patch

        apply_patch()
    except Exception as e:
        import sys

        print(f"[sitecustomize] debug metrics patch skipped: {e}", file=sys.stderr)


def _maybe_patch_verl_lora_only_ckpt() -> None:
    v = os.environ.get("PINCHBENCH_LORA_ONLY_CKPT", "").strip().lower()
    if v not in ("1", "true", "yes", "on"):
        return
    try:
        from rl.verl_lora_only_ckpt_patch import apply_patch

        apply_patch()
    except Exception as e:
        import sys

        print(f"[sitecustomize] LoRA-only checkpoint patch skipped: {e}", file=sys.stderr)


def _maybe_patch_qwen3_5_config_alias() -> None:
    """Make Transformers 4.x route qwen3_5 checkpoints through the local compatibility shim."""
    try:
        from rl.transformers_qwen3_5_patch import apply_patch

        apply_patch()
    except Exception as e:
        import sys

        print(f"[sitecustomize] qwen3_5 config alias skipped: {e}", file=sys.stderr)


def _maybe_patch_qwen3_5_generation_config() -> None:
    try:
        from rl.verl_qwen3_5_generation_patch import apply_patch

        apply_patch()
    except Exception as e:
        import sys

        print(f"[sitecustomize] qwen3_5 generation config patch skipped: {e}", file=sys.stderr)


def _maybe_patch_vllm_lora_empty_guard() -> None:
    try:
        from rl.verl_vllm_lora_empty_guard_patch import apply_patch

        apply_patch()
    except Exception as e:
        import sys

        print(f"[sitecustomize] vllm lora empty guard patch skipped: {e}", file=sys.stderr)


_maybe_patch_verl_debug_metrics()
_maybe_patch_verl_best_ckpt()
_maybe_patch_verl_lora_only_ckpt()
_maybe_patch_qwen3_5_config_alias()
_maybe_patch_qwen3_5_generation_config()
_maybe_patch_vllm_lora_empty_guard()
