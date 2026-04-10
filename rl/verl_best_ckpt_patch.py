"""
Monkey-patch veRL RayPPOTrainer._validate to prune global_step_* under trainer.default_local_dir.

After each validation, compares val-core/*/reward/mean@* (higher = better).

- New best: delete all other global_step_*; keep only global_step_{step}.
- Not best (default PINCHBENCH_KEEP_LATEST_CKPT=1): keep **both** global_step_{best_step}
  and global_step_{step} (latest save); delete any other step dirs. Does **not** delete
  the current step's directory.
- Not best (PINCHBENCH_KEEP_LATEST_CKPT=0): legacy — delete global_step_{current} only.

best_ckpt_state.json includes: best_val, best_step, latest_step.
latest_checkpointed_iteration.txt is set to **latest_step** (most recent kept checkpoint).

Requires PINCHBENCH_BEST_CKPT=1 and PYTHONPATH including repo root (for sitecustomize).
Save should run on the same steps as validation (e.g. save_freq == test_freq) so each
val has a fresh checkpoint directory to judge.
"""

from __future__ import annotations

import json
import os
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

    keep_latest = os.environ.get("PINCHBENCH_KEEP_LATEST_CKPT", "1") == "1"
    tracker = root / "latest_checkpointed_iteration.txt"

    def _iter_step_dirs():
        for p in root.iterdir():
            if not p.is_dir() or not p.name.startswith("global_step_"):
                continue
            suffix = p.name[len("global_step_") :]
            if suffix.isdigit():
                yield p

    def _write_state(bv: float, bs: int, ls: int) -> None:
        state_path.write_text(
            json.dumps(
                {"best_val": bv, "best_step": bs, "latest_step": ls},
                indent=2,
            ),
            encoding="utf-8",
        )
        if tracker.is_file():
            tracker.write_text(str(ls), encoding="utf-8")

    if val_score > best_val:
        best_val = val_score
        best_step = step
        latest_step = step
        removed = 0
        for p in _iter_step_dirs():
            if p.name == f"global_step_{step}":
                continue
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
        _write_state(best_val, best_step, latest_step)
        print(
            f"[pinchbench_best_ckpt] new best val={best_val:.4f} step={step}; "
            f"removed {removed} older checkpoint dir(s)"
        )
    else:
        if not keep_latest:
            shutil.rmtree(cur_dir, ignore_errors=True)
            if best_step is not None and (root / f"global_step_{best_step}").is_dir():
                if tracker.is_file():
                    tracker.write_text(str(best_step), encoding="utf-8")
            print(
                f"[pinchbench_best_ckpt] val={val_score:.4f} <= best={best_val:.4f}; "
                f"removed global_step_{step} only (PINCHBENCH_KEEP_LATEST_CKPT=0)"
            )
            return

        if best_step is None:
            best_step = step
        latest_step = step
        to_keep = {best_step, step}
        removed = 0
        for p in _iter_step_dirs():
            suffix = p.name[len("global_step_") :]
            if not suffix.isdigit():
                continue
            num = int(suffix)
            if num in to_keep:
                continue
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
        _write_state(best_val, best_step, latest_step)
        print(
            f"[pinchbench_best_ckpt] val={val_score:.4f} <= best={best_val:.4f}; "
            f"kept global_step_{best_step} (best) + global_step_{step} (latest); "
            f"removed {removed} other dir(s)"
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
    print("[pinchbench_best_ckpt] RayPPOTrainer._validate patched (keep best + optional latest)")
