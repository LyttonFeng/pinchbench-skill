"""
Monkey-patch veRL RayPPOTrainer._validate to keep only the best checkpoint on disk.

After each validation, compares val-core/*/reward/mean@* (higher = better).
- If this step's val is new best: delete all other global_step_* under trainer.default_local_dir.
- Else: delete global_step_{current} only (the save from this step was not an improvement).

Requires PINCHBENCH_BEST_CKPT=1 and PYTHONPATH including repo root (for sitecustomize).
Save should run on the same steps as validation (e.g. save_freq == test_freq) so each
val has a fresh checkpoint directory to judge.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _extract_val_reward(val_metrics: dict) -> float | None:
    for k, v in val_metrics.items():
        if not k.startswith("val-core/"):
            continue
        if "/reward/" not in k:
            continue
        if "mean" not in k:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _prune_checkpoints_keep_best(trainer, val_metrics: dict) -> None:
    default_dir = trainer.config.trainer.default_local_dir
    if not default_dir:
        return
    root = Path(default_dir).resolve()
    state_path = root / "best_ckpt_state.json"

    val_score = _extract_val_reward(val_metrics)
    if val_score is None:
        print("[pinchbench_best_ckpt] no val-core reward metric; skip checkpoint prune")
        return

    step = int(trainer.global_steps)
    cur_dir = root / f"global_step_{step}"
    if not cur_dir.is_dir():
        # save_freq != test_freq: no new checkpoint this step
        return

    best_val = float("-inf")
    best_step: int | None = None
    if state_path.is_file():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
            best_val = float(st.get("best_val", "-inf"))
            bs = st.get("best_step")
            if bs is not None:
                best_step = int(bs)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    tracker = root / "latest_checkpointed_iteration.txt"

    def _iter_step_dirs():
        for p in root.iterdir():
            if not p.is_dir() or not p.name.startswith("global_step_"):
                continue
            suffix = p.name[len("global_step_") :]
            if suffix.isdigit():
                yield p

    if val_score > best_val:
        best_val = val_score
        best_step = step
        removed = 0
        for p in _iter_step_dirs():
            if p.name == f"global_step_{step}":
                continue
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
        state_path.write_text(
            json.dumps({"best_val": best_val, "best_step": best_step}, indent=2),
            encoding="utf-8",
        )
        if tracker.is_file():
            tracker.write_text(str(best_step), encoding="utf-8")
        print(
            f"[pinchbench_best_ckpt] new best val={best_val:.4f} step={step}; "
            f"removed {removed} older checkpoint dir(s)"
        )
    else:
        shutil.rmtree(cur_dir, ignore_errors=True)
        if best_step is not None and (root / f"global_step_{best_step}").is_dir():
            if tracker.is_file():
                tracker.write_text(str(best_step), encoding="utf-8")
        print(
            f"[pinchbench_best_ckpt] val={val_score:.4f} <= best={best_val:.4f}; "
            f"removed global_step_{step} only"
        )


def apply_patch() -> None:
    from verl.trainer.ppo import ray_trainer as rt

    if getattr(rt.RayPPOTrainer, "_pinchbench_best_ckpt_patched", False):
        return

    _orig = rt.RayPPOTrainer._validate

    def _validate(self, merged: bool = False):
        out = _orig(self, merged=merged)
        if not merged:
            try:
                _prune_checkpoints_keep_best(self, out)
            except Exception as e:
                print(f"[pinchbench_best_ckpt] prune failed: {e}")
        return out

    rt.RayPPOTrainer._validate = _validate  # type: ignore[method-assign]
    rt.RayPPOTrainer._pinchbench_best_ckpt_patched = True
    print("[pinchbench_best_ckpt] RayPPOTrainer._validate patched (keep-best checkpoints)")
