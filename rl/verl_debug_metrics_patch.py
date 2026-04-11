"""
Patch veRL calculate_debug_metrics when rollout_log_probs vs old_log_probs yields an empty
masked tensor — torch.max(empty) raises RuntimeError and kills training.

Upstream fix (newer verl): guard when response_mask is all False; some versions still hit
empty masked_select in edge cases (proxy timeout, aborted rollout, shape quirks).

Set PINCHBENCH_DEBUG_METRICS_PATCH=0 to disable.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def apply_patch() -> object:
    import verl.utils.debug.metrics as m

    if getattr(m, "_pinchbench_debug_metrics_patched", False):
        return m.calculate_debug_metrics

    _orig = m.calculate_debug_metrics

    def calculate_debug_metrics_safe(data):
        try:
            return _orig(data)
        except RuntimeError as e:
            msg = str(e).lower()
            if "numel() == 0" in msg or (
                "max()" in msg and "reduction dim" in msg and "specified" in msg
            ):
                logger.warning(
                    "[pinchbench] calculate_debug_metrics: empty diff tensor (%s); "
                    "returning default debug metrics",
                    e,
                )
                return {
                    "training/rollout_probs_diff_valid": 0,
                    "training/rollout_probs_diff_max": float("nan"),
                    "training/rollout_probs_diff_mean": float("nan"),
                    "training/rollout_probs_diff_std": float("nan"),
                    "training/rollout_actor_probs_pearson_corr": float("nan"),
                }
            raise

    m.calculate_debug_metrics = calculate_debug_metrics_safe
    m._pinchbench_debug_metrics_patched = True

    # Some verl versions import the function into ray_trainer as a local symbol.
    # Patch that alias too so the trainer sees the safe wrapper instead of the
    # original function that may still call torch.max(empty).
    try:
        import verl.trainer.ppo.ray_trainer as ray_trainer

        ray_trainer.calculate_debug_metrics = calculate_debug_metrics_safe
    except Exception as e:
        logger.debug("[pinchbench] ray_trainer alias patch skipped: %s", e)

    for module_name in ("verl.trainer.main_ppo",):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "calculate_debug_metrics"):
            try:
                setattr(module, "calculate_debug_metrics", calculate_debug_metrics_safe)
            except Exception as e:
                logger.debug(
                    "[pinchbench] %s alias patch skipped: %s",
                    module_name,
                    e,
                )

    print(
        "[pinchbench] verl.utils.debug.metrics.calculate_debug_metrics patched "
        "(safe empty rollout_probs_diff)"
    )
    return calculate_debug_metrics_safe
