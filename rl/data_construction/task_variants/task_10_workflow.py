"""task_10_workflow variant generator — varies config.json endpoint and task description."""
from __future__ import annotations
import json
import random
from typing import Iterator
from . import BaseVariantGenerator, TaskVariant

_CONFIGS = [
    {
        "endpoint": "https://api.example.com/v1/users",
        "method": "GET",
        "auth": "Bearer token",
        "description": "Fetch user list",
        "script_name": "fetch_users.py",
        "notes_name": "NOTES.md",
    },
    {
        "endpoint": "https://api.payments.io/v2/transactions",
        "method": "POST",
        "auth": "API-Key",
        "description": "Submit payment transaction",
        "script_name": "submit_payment.py",
        "notes_name": "NOTES.md",
    },
    {
        "endpoint": "https://data.internal.corp/metrics/daily",
        "method": "GET",
        "auth": "OAuth2",
        "description": "Pull daily metrics",
        "script_name": "pull_metrics.py",
        "notes_name": "NOTES.md",
    },
    {
        "endpoint": "https://notify.service.net/webhooks/send",
        "method": "POST",
        "auth": "HMAC-SHA256",
        "description": "Send webhook notification",
        "script_name": "send_webhook.py",
        "notes_name": "NOTES.md",
    },
    {
        "endpoint": "https://search.api.co/v3/documents",
        "method": "GET",
        "auth": "Bearer token",
        "description": "Search documents",
        "script_name": "search_docs.py",
        "notes_name": "NOTES.md",
    },
]

_PROMPT_TEMPLATE = (
    "Read config.json, extract the API endpoint, create a Python script to call it, "
    "and document the process in {notes_name}."
)


class Generator(BaseVariantGenerator):
    task_id = "task_10_workflow"

    def sample(self, n: int = 1, seed: int | None = None) -> Iterator[TaskVariant]:
        rng = random.Random(seed)
        pool = _CONFIGS.copy()
        rng.shuffle(pool)
        for i in range(n):
            cfg = pool[i % len(pool)]
            config_json = {
                "endpoint": cfg["endpoint"],
                "method": cfg["method"],
                "auth_type": cfg["auth"],
                "timeout": rng.choice([10, 30, 60]),
                "retry": rng.choice([1, 2, 3]),
            }
            yield TaskVariant(
                task_id=self.task_id,
                prompt=_PROMPT_TEMPLATE.format(notes_name=cfg["notes_name"]),
                workspace_files={
                    "config.json": json.dumps(config_json, indent=2),
                },
                metadata={"endpoint": cfg["endpoint"], "description": cfg["description"]},
            )
