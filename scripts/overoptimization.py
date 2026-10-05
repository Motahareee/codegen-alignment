"""Reward over-optimization curves from GRPO/RLHF training logs.

For each run (a checkpoint dir with log_history.json from train/grpo.py --log-kl),
show how the proxy reward (what RL optimizes), the gold reward (tests passed,
never trained on in RLHF runs) and the KL from the starting model evolve.
Gao et al. 2022 plot rewards against sqrt(KL): the proxy keeps rising while the
gold reward rises, peaks, then falls once the policy exploits the reward model.

Writes a markdown table (steps averaged into bins) and, if matplotlib is
installed, a plot next to it.

  python scripts/overoptimization.py checkpoints/rlhf_*_kodcode --out outputs/overoptimization_kodcode.md
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_run(path: Path):
    logs = [r for r in json.load(open(path / "log_history.json")) if "kl" in r]
    proxy_key = next((k for k in logs[0] if k.startswith("rewards/") and k.endswith("/mean")
                      and "test_gold" not in k), None)
    gold_key = "rewards/test_gold/mean" if "rewards/test_gold/mean" in logs[0] else proxy_key  # --reward binary
    col = lambda k: np.array([r[k] for r in logs], dtype=float)
    return {"step": col("step"), "kl": col("kl"), "proxy": col(proxy_key), "gold": col(gold_key),
            "proxy_name": proxy_key.split("/")[1]}


def binned(x, n_bins):
    return [float(np.mean(b)) for b in np.array_split(x, n_bins)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="checkpoint dirs containing log_history.json")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", default="outputs/overoptimization.md")
    args = ap.parse_args()

    runs = {Path(r).name: load_run(Path(r)) for r in args.runs if (Path(r) / "log_history.json").exists()}
    L = ["# Reward over-optimization", "",
         "*proxy* = the reward RL optimizes (reward model score, or tests for test-reward runs); "
         "*gold* = fraction of sampled training answers that pass all tests; "
         "*sqrt KL* = distance from the starting model. Each row averages 1/%d of the training steps." % args.bins, ""]
    for name, r in runs.items():
        L += [f"## {name} (proxy: {r['proxy_name']})", "",
              "| steps | sqrt KL | proxy | gold |", "|---|---|---|---|"]
        steps = np.array_split(r["step"], args.bins)
        for s, kl, p, g in zip(steps, binned(r["kl"], args.bins), binned(r["proxy"], args.bins), binned(r["gold"], args.bins)):
            L.append(f"| {int(s[0])}-{int(s[-1])} | {np.sqrt(max(kl, 0)):.3f} | {p:+.3f} | {g:.3f} |")
        L.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"-> {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: table only)")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, r in runs.items():
        x = np.sqrt(np.clip(binned(r["kl"], args.bins), 0, None))
        ax1.plot(x, binned(r["gold"], args.bins), marker="o", label=name)
        ax2.plot(x, binned(r["proxy"], args.bins), marker="o", label=name)
    ax1.set(xlabel="sqrt KL from start", ylabel="gold: pass rate on training samples", title="Gold reward (tests)")
    ax2.set(xlabel="sqrt KL from start", ylabel="proxy reward", title="Proxy reward (what RL optimizes)")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"-> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
