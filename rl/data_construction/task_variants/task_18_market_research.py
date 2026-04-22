"""task_18_market_research variant generator — varies market segment for research."""
from __future__ import annotations
import random
from typing import Iterator
from . import BaseVariantGenerator, TaskVariant

_SEGMENTS = [
    {
        "name": "apm_observability",
        "market": "enterprise observability and APM (Application Performance Monitoring)",
        "competitors": "Datadog, New Relic, Dynatrace, Splunk, Grafana Labs",
        "output_file": "market_research.md",
    },
    {
        "name": "cloud_security",
        "market": "cloud security and CSPM (Cloud Security Posture Management)",
        "competitors": "Wiz, Orca Security, Palo Alto Networks Prisma, CrowdStrike, Lacework",
        "output_file": "market_research.md",
    },
    {
        "name": "developer_tools",
        "market": "developer productivity and CI/CD tooling",
        "competitors": "GitHub Actions, GitLab CI, CircleCI, JetBrains TeamCity, Buildkite",
        "output_file": "market_research.md",
    },
    {
        "name": "vector_databases",
        "market": "vector databases and AI-native data infrastructure",
        "competitors": "Pinecone, Weaviate, Qdrant, Milvus, Chroma",
        "output_file": "market_research.md",
    },
    {
        "name": "api_management",
        "market": "API management and gateway platforms",
        "competitors": "Kong, Apigee (Google), AWS API Gateway, MuleSoft, Tyk",
        "output_file": "market_research.md",
    },
    {
        "name": "data_orchestration",
        "market": "data orchestration and pipeline management",
        "competitors": "Apache Airflow, Prefect, Dagster, Astronomer, Mage",
        "output_file": "market_research.md",
    },
    {
        "name": "feature_stores",
        "market": "ML feature stores and MLOps platforms",
        "competitors": "Feast, Tecton, Hopsworks, AWS SageMaker Feature Store, Vertex AI",
        "output_file": "market_research.md",
    },
    {
        "name": "identity_access",
        "market": "identity and access management (IAM) for enterprise",
        "competitors": "Okta, Microsoft Entra ID, Ping Identity, Auth0, CyberArk",
        "output_file": "market_research.md",
    },
]

_PROMPT_TEMPLATE = (
    "Create a competitive landscape analysis for the **{market}** market segment. "
    "Based on your knowledge, identify the top 5 players (key ones include: {competitors}), "
    "their key differentiators, market trends, and typical pricing models. "
    "Save your findings to a file called `{output_file}` in a well-structured format "
    "with sections for each competitor and a summary comparison table.\n\n"
    "If you have access to web search tools, use them to gather the most current information. "
    "Otherwise, use your knowledge of this market to produce a thorough analysis."
)


class Generator(BaseVariantGenerator):
    task_id = "task_18_market_research"

    def sample(self, n: int = 1, seed: int | None = None) -> Iterator[TaskVariant]:
        rng = random.Random(seed)
        pool = _SEGMENTS.copy()
        rng.shuffle(pool)
        for i in range(n):
            seg = pool[i % len(pool)]
            yield TaskVariant(
                task_id=self.task_id,
                prompt=_PROMPT_TEMPLATE.format(
                    market=seg["market"],
                    competitors=seg["competitors"],
                    output_file=seg["output_file"],
                ),
                workspace_files={},
                metadata={"segment": seg["name"]},
            )
