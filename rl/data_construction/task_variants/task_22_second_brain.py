"""task_22_second_brain variant generator — varies the personal facts to store/recall."""
from __future__ import annotations
import random
from typing import Iterator
from . import BaseVariantGenerator, TaskVariant

_PROFILES = [
    {
        "name": "rust_developer",
        "language": "Rust",
        "start_date": "January 15, 2024",
        "mentor_name": "Dr. Elena Vasquez",
        "mentor_affiliation": "Stanford",
        "project_name": "NeonDB",
        "project_desc": "a distributed key-value store",
        "secret_phrase": "purple elephant sunrise",
    },
    {
        "name": "go_developer",
        "language": "Go",
        "start_date": "March 8, 2023",
        "mentor_name": "Prof. James Hartley",
        "mentor_affiliation": "MIT",
        "project_name": "SwiftRouter",
        "project_desc": "a high-performance HTTP load balancer",
        "secret_phrase": "crimson falcon orbit",
    },
    {
        "name": "python_ml",
        "language": "Python",
        "start_date": "September 22, 2023",
        "mentor_name": "Dr. Priya Sharma",
        "mentor_affiliation": "Carnegie Mellon University",
        "project_name": "NebulaML",
        "project_desc": "an automated machine learning pipeline framework",
        "secret_phrase": "silver tortoise midnight",
    },
    {
        "name": "kotlin_android",
        "language": "Kotlin",
        "start_date": "June 1, 2024",
        "mentor_name": "Dr. Yuki Tanaka",
        "mentor_affiliation": "Tokyo Institute of Technology",
        "project_name": "AuraSync",
        "project_desc": "a real-time collaborative note-taking Android app",
        "secret_phrase": "golden phoenix delta",
    },
    {
        "name": "elixir_backend",
        "language": "Elixir",
        "start_date": "November 11, 2023",
        "mentor_name": "Prof. Sofia Moreau",
        "mentor_affiliation": "EPFL",
        "project_name": "IronMesh",
        "project_desc": "a fault-tolerant message broker built on OTP",
        "secret_phrase": "blue jaguar cascade",
    },
    {
        "name": "zig_systems",
        "language": "Zig",
        "start_date": "February 28, 2024",
        "mentor_name": "Dr. Marcus Webb",
        "mentor_affiliation": "ETH Zurich",
        "project_name": "VortexOS",
        "project_desc": "a microkernel operating system for embedded devices",
        "secret_phrase": "amber wolf labyrinth",
    },
]

_SESSION1_TEMPLATE = (
    "I want you to remember this important information for me. "
    "Please save it to a file called `memory/MEMORY.md` so you (or a future session) can recall it later:\n\n"
    "My favorite programming language is {language}. I started learning it on {start_date}. "
    "My mentor's name is {mentor_name} from {mentor_affiliation}. "
    "The project I'm working on is called \"{project_name}\" - it's {project_desc}. "
    "The secret code phrase for our team is \"{secret_phrase}\".\n\n"
    "Save this information to `memory/MEMORY.md` and confirm you've stored it."
)

_SESSION2_TEMPLATE = (
    "What programming language am I learning? And what's the name of my current project? "
    "You can check the memory/MEMORY.md file if needed."
)

_SESSION3_TEMPLATE = (
    "I previously saved some personal information in a file called `memory/MEMORY.md`. "
    "Please read that file and answer these questions:\n"
    "1. What is my favorite programming language?\n"
    "2. When did I start learning it?\n"
    "3. What is my mentor's name and affiliation?\n"
    "4. What is my project called and what does it do?\n"
    "5. What is my team's secret code phrase?\n\n"
    "Please answer all 5 questions based on what you find in the file."
)


class Generator(BaseVariantGenerator):
    task_id = "task_22_second_brain"

    def sample(self, n: int = 1, seed: int | None = None) -> Iterator[TaskVariant]:
        rng = random.Random(seed)
        pool = _PROFILES.copy()
        rng.shuffle(pool)
        for i in range(n):
            p = pool[i % len(pool)]
            # The benchmark is multi-session; for SFT data we use session 1 (store) as the primary prompt
            # Include all session prompts in metadata so build_dataset.py can handle multi-session runs
            yield TaskVariant(
                task_id=self.task_id,
                prompt=_SESSION1_TEMPLATE.format(**p),
                workspace_files={},
                metadata={
                    "profile": p["name"],
                    "session2_prompt": _SESSION2_TEMPLATE,
                    "session3_prompt": _SESSION3_TEMPLATE,
                    "facts": {
                        "language": p["language"],
                        "start_date": p["start_date"],
                        "mentor": f"{p['mentor_name']} from {p['mentor_affiliation']}",
                        "project": p["project_name"],
                        "project_desc": p["project_desc"],
                        "secret_phrase": p["secret_phrase"],
                    },
                },
            )
