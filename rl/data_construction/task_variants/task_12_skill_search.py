"""task_12_skill_search variant generator.

Varies the config files content and the search-replace rules.
The key skill being tested: read files first, then edit — don't guess filenames.
"""
from __future__ import annotations
import json
import random
import yaml  # pyyaml
from typing import Iterator
from . import BaseVariantGenerator, TaskVariant

# Each scenario defines what files exist and what replacements to make
_SCENARIOS = [
    {
        "name": "db_migration",
        "prompt": (
            "Update the configuration files for production deployment:\n"
            "1. In config/settings.json: change the log level from \"debug\" to \"warn\"\n"
            "2. In config/database.yml: change the host from \"localhost\" to \"prod-db.example.com\" "
            "and the database name from \"myapp_dev\" to \"myapp_prod\""
        ),
        "files": {
            "config/settings.json": lambda rng: json.dumps({
                "database": {"host": "localhost", "port": 5432, "name": "myapp_dev"},
                "api": {"endpoint": "http://localhost:3000", "timeout": 30},
                "logging": {"level": "debug", "file": "/var/log/myapp/dev.log"},
            }, indent=2),
            "config/database.yml": lambda rng: yaml.dump({
                "default": {"adapter": "postgresql", "host": "localhost",
                            "database": "myapp_dev", "username": "devuser",
                            "password": "dev_password_123", "pool": 5},
            }),
        },
    },
    {
        "name": "api_migration",
        "prompt": (
            "Update config files for staging environment:\n"
            "1. In config/app.json: change \"environment\" from \"development\" to \"staging\"\n"
            "2. In config/services.yml: change the base_url from \"http://localhost:8080\" "
            "to \"https://staging.api.company.com\""
        ),
        "files": {
            "config/app.json": lambda rng: json.dumps({
                "environment": "development",
                "debug": True,
                "version": f"1.{rng.randint(0,9)}.{rng.randint(0,9)}",
                "max_connections": rng.choice([10, 20, 50]),
            }, indent=2),
            "config/services.yml": lambda rng: yaml.dump({
                "payment": {"base_url": "http://localhost:8080", "timeout": 30, "retry": 3},
                "notification": {"base_url": "http://localhost:8080", "timeout": 10},
            }),
        },
    },
    {
        "name": "feature_flags",
        "prompt": (
            "Update feature flag configuration:\n"
            "1. In config/features.json: change \"new_dashboard\" from false to true "
            "and \"legacy_api\" from true to false\n"
            "2. In config/rollout.yml: change the rollout percentage from 0 to 100"
        ),
        "files": {
            "config/features.json": lambda rng: json.dumps({
                "new_dashboard": False,
                "legacy_api": True,
                "beta_search": rng.choice([True, False]),
                "dark_mode": True,
            }, indent=2),
            "config/rollout.yml": lambda rng: yaml.dump({
                "new_dashboard": {"enabled": True, "percentage": 0, "groups": ["beta"]},
                "legacy_api": {"enabled": True, "percentage": 100},
            }),
        },
    },
    {
        "name": "cache_config",
        "prompt": (
            "Update cache configuration for production:\n"
            "1. In config/cache.json: change \"backend\" from \"memory\" to \"redis\" "
            "and \"ttl\" from 60 to 3600\n"
            "2. In config/redis.yml: change host from \"localhost\" to \"redis.prod.internal\""
        ),
        "files": {
            "config/cache.json": lambda rng: json.dumps({
                "backend": "memory",
                "ttl": 60,
                "max_size": rng.choice([100, 500, 1000]),
                "eviction_policy": "lru",
            }, indent=2),
            "config/redis.yml": lambda rng: yaml.dump({
                "redis": {"host": "localhost", "port": 6379,
                          "db": rng.randint(0, 3), "password": None},
            }),
        },
    },
    {
        "name": "logging_config",
        "prompt": (
            "Update logging settings:\n"
            "1. In config/logging.json: change \"level\" from \"info\" to \"error\" "
            "and \"console_output\" from true to false\n"
            "2. In config/handlers.yml: change the log file path from "
            "\"/tmp/app.log\" to \"/var/log/production/app.log\""
        ),
        "files": {
            "config/logging.json": lambda rng: json.dumps({
                "level": "info",
                "console_output": True,
                "structured": rng.choice([True, False]),
                "max_file_size_mb": rng.choice([10, 50, 100]),
            }, indent=2),
            "config/handlers.yml": lambda rng: yaml.dump({
                "file_handler": {"path": "/tmp/app.log", "rotate": True, "max_files": 7},
                "syslog": {"enabled": False},
            }),
        },
    },
]


class Generator(BaseVariantGenerator):
    task_id = "task_12_skill_search"

    def sample(self, n: int = 1, seed: int | None = None) -> Iterator[TaskVariant]:
        rng = random.Random(seed)
        pool = _SCENARIOS.copy()
        rng.shuffle(pool)
        for i in range(n):
            scenario = pool[i % len(pool)]
            workspace_files = {
                path: gen(rng) for path, gen in scenario["files"].items()
            }
            yield TaskVariant(
                task_id=self.task_id,
                prompt=scenario["prompt"],
                workspace_files=workspace_files,
                metadata={"scenario": scenario["name"]},
            )
