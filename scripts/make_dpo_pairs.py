"""Build DPO preference pairs from the model's own samples.

Input is the samples.jsonl written by evaluate.py run on the *train* split
with several samples per problem, e.g.:

  python scripts/evaluate.py --model checkpoints/sft --chat --split train --n 8 --temperature 1.0

For each problem, a passing completion is "chosen" and a failing one is
"rejected". Problems where all samples pass (or all fail) give no signal and
are skipped: that is exactly the "learnable" zone GRPO needs too.
"""

import argparse
import json
import random
from pathlib import Path

from codealign.data import load_split

ap = argparse.ArgumentParser()
ap.add_argument("samples", help="samples.jsonl from evaluate.py --split train")
ap.add_argument("--out", default="data/dpo_pairs.jsonl")
ap.add_argument("--max-pairs-per-problem", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = random.Random(args.seed)
problems = {p.task_id: p for p in load_split("train")}
pairs, n_problems, n_all_pass, n_all_fail = [], 0, 0, 0

for line in open(args.samples):
    row = json.loads(line)
    n_problems += 1
    good = [c for c, k in zip(row["completions"], row["passed"]) if k == row["total"]]
    bad = [c for c, k in zip(row["completions"], row["passed"]) if k < row["total"]]
    if not bad:
        n_all_pass += 1
        continue
    if not good:
        n_all_fail += 1
        continue
    rng.shuffle(good)
    rng.shuffle(bad)
    messages = problems[row["task_id"]].messages()
    for chosen, rejected in list(zip(good, bad))[: args.max_pairs_per_problem]:
        pairs.append({
            "task_id": row["task_id"],
            "prompt": messages,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

useful = n_problems - n_all_pass - n_all_fail
print(f"{n_problems} problems: {n_all_pass} all-pass, {n_all_fail} all-fail, {useful} mixed")
print(f"wrote {len(pairs)} pairs -> {args.out}")
