"""Build SFT data from a teacher's own answers that pass the tests (sequence-level distillation).

Input is the samples.jsonl written by evaluate.py on the KodCode split, e.g.:

  python scripts/evaluate.py --model checkpoints/grpo_kodcode_1.5b --chat --split kodcode --limit 20000 --n 4 --temperature 0.7
  python scripts/make_distill_data.py outputs/eval/grpo_kodcode_1.5b_kodcode_n4_t0.7/samples.jsonl --out data/kd_seq.jsonl

The student is then fine-tuned on these rows with train/sft.py --data data/kd_seq.jsonl.
Only passing answers are kept (rejection sampling), so the student learns the
teacher's style and solutions but not its mistakes. Problems the teacher never
solved are dropped.
"""

import argparse
import json
from pathlib import Path

from codealign.data import SYSTEM_PROMPT

ap = argparse.ArgumentParser()
ap.add_argument("samples", help="samples.jsonl from evaluate.py --split kodcode")
ap.add_argument("--out", default="data/kd_seq.jsonl")
ap.add_argument("--per-problem", type=int, default=1, help="max distinct passing answers kept per problem")
args = ap.parse_args()

rows, n_problems, n_solved = [], 0, 0
for line in open(args.samples):
    row = json.loads(line)
    n_problems += 1
    good = list(dict.fromkeys(  # distinct, in sample order
        c for c, k in zip(row["completions"], row["passed"]) if k == row["total"]))
    n_solved += bool(good)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["prompt"]}]
    for answer in good[: args.per_problem]:
        rows.append({"prompt": messages, "completion": [{"role": "assistant", "content": answer.strip()}]})

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"{n_problems} problems, teacher solved {n_solved} ({n_solved / max(n_problems, 1):.0%})")
print(f"wrote {len(rows)} rows -> {args.out}")
