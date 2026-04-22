"""
Task variant generators — each produces (prompt, workspace_files) pairs for SFT/DPO data construction.

Usage:
    from rl.data_construction.task_variants import get_generator
    gen = get_generator("task_02_stock")
    for variant in gen.sample(n=10):
        print(variant.prompt)
        print(variant.workspace_files)  # dict[filename -> bytes | str]
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator
import importlib

@dataclass
class TaskVariant:
    task_id: str
    prompt: str
    workspace_files: dict[str, str | bytes] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)  # e.g. {"ticker": "GOOGL", "difficulty": "medium"}
    expected: dict = field(default_factory=dict)


class BaseVariantGenerator:
    task_id: str

    def sample(self, n: int = 1, seed: int | None = None) -> Iterator[TaskVariant]:
        """Yield n distinct variants."""
        raise NotImplementedError


_REGISTRY: dict[str, str] = {
    "task_02_stock":               "rl.data_construction.task_variants.task_02_stock",
    "task_10_workflow":            "rl.data_construction.task_variants.task_10_workflow",
    "task_12_skill_search":        "rl.data_construction.task_variants.task_12_skill_search",
    "task_16_email_triage":        "rl.data_construction.task_variants.task_16_email_triage",
    "task_18_market_research":     "rl.data_construction.task_variants.task_18_market_research",
    "task_18_spreadsheet_summary": "rl.data_construction.task_variants.task_18_spreadsheet_summary",
    "task_22_second_brain":        "rl.data_construction.task_variants.task_22_second_brain",
    "task_24_polymarket_briefing": "rl.data_construction.task_variants.task_24_polymarket_briefing",
}


def get_generator(task_id: str) -> BaseVariantGenerator:
    module_path = _REGISTRY[task_id]
    mod = importlib.import_module(module_path)
    return mod.Generator()
