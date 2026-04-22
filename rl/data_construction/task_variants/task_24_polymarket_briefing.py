"""task_24_polymarket_briefing variant generator — varies date framing only (live task)."""
from __future__ import annotations
import random
from typing import Iterator
from . import BaseVariantGenerator, TaskVariant

# The task is inherently live (fetches real-time Polymarket data).
# We vary the prompt framing slightly — number of markets, output file name,
# and whether to include volume info — to create surface diversity while
# keeping the core fetch-and-summarize skill the same.
_VARIANTS = [
    {
        "name": "top3_standard",
        "n_markets": 3,
        "output_file": "polymarket_briefing.md",
        "extra": "",
    },
    {
        "name": "top5_extended",
        "n_markets": 5,
        "output_file": "polymarket_briefing.md",
        "extra": " Also include the 24-hour volume for each market.",
    },
    {
        "name": "top3_with_volume",
        "n_markets": 3,
        "output_file": "polymarket_briefing.md",
        "extra": " For each market, include the total liquidity and 24h volume if available.",
    },
    {
        "name": "top3_politics_focus",
        "n_markets": 3,
        "output_file": "polymarket_briefing.md",
        "extra": " Prefer political or macroeconomic markets if multiple categories are trending.",
    },
    {
        "name": "top3_tech_focus",
        "n_markets": 3,
        "output_file": "polymarket_briefing.md",
        "extra": " Prefer technology or AI-related markets if multiple categories are trending.",
    },
]

_PROMPT_TEMPLATE = (
    "Fetch the top {n_markets} trending prediction markets from Polymarket (polymarket.com) right now. "
    "For each market, find a related recent news story (from the last 48 hours) that explains why people are betting on it.{extra}\n\n"
    "Save the result as `{output_file}` with the format:\n\n"
    "```\n"
    "# Polymarket Briefing — {{today's date}}\n\n"
    "## 1. {{Market Question}}\n"
    "**Current odds:** Yes {{X}}% / No {{Y}}%\n"
    "**Related news:** {{1-2 sentence summary of a real news story that contextualizes this market}}\n\n"
    "## 2. {{Market Question}}\n"
    "**Current odds:** Yes {{X}}% / No {{Y}}%\n"
    "**Related news:** {{1-2 sentence summary}}\n"
    "```\n\n"
    "(Continue for all {n_markets} markets)\n\n"
    "Only use real, currently active markets. Do not fabricate markets or odds.\n\n"
    "API reference: `https://gamma-api.polymarket.com/markets?active=true&order=volumeNum&ascending=false&limit=10`"
)


class Generator(BaseVariantGenerator):
    task_id = "task_24_polymarket_briefing"

    def sample(self, n: int = 1, seed: int | None = None) -> Iterator[TaskVariant]:
        rng = random.Random(seed)
        pool = _VARIANTS.copy()
        rng.shuffle(pool)
        for i in range(n):
            v = pool[i % len(pool)]
            yield TaskVariant(
                task_id=self.task_id,
                prompt=_PROMPT_TEMPLATE.format(
                    n_markets=v["n_markets"],
                    output_file=v["output_file"],
                    extra=v["extra"],
                ),
                workspace_files={},
                metadata={"variant": v["name"]},
            )
