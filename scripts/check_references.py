"""Sanity check: gold solutions should pass their own tests in our sandbox.

Anything that fails here is a data or sandbox problem, not a model problem,
and should be excluded from rewards and evaluation.
"""

import argparse
from collections import Counter

from codealign.data import load_split
from codealign.sandbox import run_many

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="eval", choices=["train", "eval"])
args = parser.parse_args()

problems = load_split(args.split)
results = run_many([{"program": p.reference, "tests": p.tests, "setup": p.setup} for p in problems])

print(f"{args.split}: {sum(r.all_passed for r in results)}/{len(problems)} reference solutions pass")
print("status counts:", dict(Counter(r.status for r in results)))
for p, r in zip(problems, results):
    if not r.all_passed:
        print(f"  {p.task_id}: {r.status} {r.passed}/{r.total} {r.detail}")
