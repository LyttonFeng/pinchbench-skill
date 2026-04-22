#!/usr/bin/env python3
"""Launch veRL main_ppo after applying local compatibility patches."""

from __future__ import annotations

import runpy

import rl.transformers_qwen3_5_patch  # noqa: F401
import rl.verl_qwen3_5_generation_patch  # noqa: F401

runpy.run_module("verl.trainer.main_ppo", run_name="__main__")
