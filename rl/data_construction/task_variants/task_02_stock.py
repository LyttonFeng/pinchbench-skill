"""task_02_stock variant generator — varies stock ticker and output filename."""
from __future__ import annotations
import random
from typing import Iterator
from . import BaseVariantGenerator, TaskVariant

# (ticker, company_name, output_file)
_STOCKS = [
    ("AAPL",  "Apple Inc.",             "stock_report.txt"),
    ("GOOGL", "Alphabet Inc. (Google)", "stock_report.txt"),
    ("MSFT",  "Microsoft Corporation",  "stock_report.txt"),
    ("TSLA",  "Tesla, Inc.",            "stock_report.txt"),
    ("AMZN",  "Amazon.com, Inc.",       "stock_report.txt"),
    ("NVDA",  "NVIDIA Corporation",     "stock_report.txt"),
    ("META",  "Meta Platforms, Inc.",   "stock_report.txt"),
    ("NFLX",  "Netflix, Inc.",          "stock_report.txt"),
    ("AMD",   "Advanced Micro Devices", "stock_report.txt"),
    ("BABA",  "Alibaba Group",          "stock_report.txt"),
    ("TSM",   "Taiwan Semiconductor",   "stock_report.txt"),
    ("BIDU",  "Baidu, Inc.",            "stock_report.txt"),
]

_PROMPT_TEMPLATE = (
    "Research the current stock price of {company} ({ticker}) and save it to "
    "{output_file} with the price, date, and a brief market summary."
)


class Generator(BaseVariantGenerator):
    task_id = "task_02_stock"

    def sample(self, n: int = 1, seed: int | None = None) -> Iterator[TaskVariant]:
        rng = random.Random(seed)
        pool = _STOCKS.copy()
        rng.shuffle(pool)
        for i in range(n):
            ticker, company, output_file = pool[i % len(pool)]
            yield TaskVariant(
                task_id=self.task_id,
                prompt=_PROMPT_TEMPLATE.format(
                    ticker=ticker, company=company, output_file=output_file
                ),
                workspace_files={},
                metadata={"ticker": ticker, "company": company},
            )
