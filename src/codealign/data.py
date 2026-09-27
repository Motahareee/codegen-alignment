"""Load coding problems into one common shape and build prompts.

Splits:
  - train: MBPP "full" train (373 usable problems, task_ids 601-974)
  - eval:  MBPP "sanitized" test (257 problems, task_ids 11-510)
They don't overlap, so we never train on eval problems.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset

SYSTEM_PROMPT = "You are an expert Python programmer."

# Problems whose gold solution fails its own tests (see scripts/check_references.py).
BROKEN = {"mbpp/927"}  # tests reference an undefined `Node` class


@dataclass
class Problem:
    task_id: str
    description: str
    tests: list[str]
    setup: str  # imports / setup code the tests need
    reference: str  # gold solution (used for SFT targets)

    @property
    def prompt(self) -> str:
        # Showing one test tells the model the expected function name and signature.
        return (
            f"{self.description}\n"
            f"Your code should pass this test:\n{self.tests[0]}\n"
            "Write the complete solution in a single ```python code block."
        )

    def messages(self) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.prompt},
        ]


def load_split(split: str) -> list[Problem]:
    if split == "train":
        ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
        problems = [
            Problem(
                task_id=f"mbpp/{ex['task_id']}",
                description=ex["text"],
                tests=ex["test_list"],
                setup=ex["test_setup_code"],
                reference=ex["code"],
            )
            for ex in ds
        ]
        return [p for p in problems if p.task_id not in BROKEN]
    if split == "eval":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        return [
            Problem(
                task_id=f"mbpp/{ex['task_id']}",
                description=ex["prompt"],
                tests=ex["test_list"],
                setup="\n".join(ex["test_imports"]),
                reference=ex["code"],
            )
            for ex in ds
        ]
    raise ValueError(f"unknown split {split!r}; expected 'train' or 'eval'")


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str:
    """Pull the code out of a model completion.

    Uses the last fenced block (models often reason first, answer last). If a
    fence was opened but never closed (hit max_tokens), take everything after
    it. With no fences, assume the whole completion is code.
    """
    blocks = _FENCE.findall(completion)
    if blocks:
        return blocks[-1]
    m = re.search(r"```(?:python|py)?[ \t]*\n", completion)
    if m:
        return completion[m.end():]
    return completion
