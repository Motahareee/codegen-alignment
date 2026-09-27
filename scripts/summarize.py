"""Collect every metrics.json under an eval dir into one comparison table."""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval")
rows = [json.loads(p.read_text()) for p in sorted(root.glob("*/metrics.json"))]
rows = [r for r in rows if r["split"] == "eval"]

by_model: dict[str, dict] = {}
for r in rows:
    m = by_model.setdefault(r["model"], {})
    if r["n"] == 1 and r["temperature"] == 0:
        m["greedy pass@1"] = r["pass@1"]
        m["test frac"] = r["mean_test_frac"]
    else:
        m["sampled pass@1"] = r["pass@1"]
        if "pass@10" in r:
            m["pass@10"] = r["pass@10"]

cols = ["greedy pass@1", "sampled pass@1", "pass@10", "test frac"]
print(f"| {'model':40} | " + " | ".join(cols) + " |")
print("|" + "-" * 42 + "|" + "|".join("-" * (len(c) + 2) for c in cols) + "|")
for model, m in by_model.items():
    cells = [f"{m[c]:.3f}".rjust(len(c)) if c in m else "-".rjust(len(c)) for c in cols]
    print(f"| {model:40} | " + " | ".join(cells) + " |")
