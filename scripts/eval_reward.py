"""Grade reward models against the unit tests, on eval samples they never saw.

Each sample in samples.jsonl (from evaluate.py on the eval split) has a known
pass/fail label. A scorer gives every sample a number; we then ask:

  1. pairwise accuracy  for a passing and a failing answer to the same problem, how often does
                        the passing one score higher? (0.5 = coin flip, 1.0 = perfect)
  2. best-of-k          pick the highest-scoring of k samples: how often does it pass?
                        Between random choice (pass@1) and a perfect picker (pass@k).
  3. length bias        within a problem, does the score just track answer length?

Scorers: a trained reward model (--rm, train/reward.py) and DPO's implicit reward
(--implicit DPO REF): log pi_dpo(y|x) - log pi_ref(y|x), the reward DPO optimizes without
ever building a reward model. Needs a GPU for speed; CPU works for --limit smoke tests.

  python scripts/eval_reward.py --rm checkpoints/rm_kodcode \
      --implicit checkpoints/dpo_kodcode checkpoints/sft_kodcode \
      --samples outputs/eval/sft_kodcode_eval_n10_t0.8/samples.jsonl --out outputs/reward_eval.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from codealign.data import SYSTEM_PROMPT
from codealign.metrics import pass_at_k

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def conversations(rows):
    """One (prompt messages, full messages) pair per sample, in row order."""
    out = []
    for r in rows:
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["prompt"]}]
        out += [(prompt, prompt + [{"role": "assistant", "content": c}]) for c in r["completions"]]
    return out


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


@torch.no_grad()
def rm_scores(path, convs, batch_size):
    """Reward model score for each full conversation (same chat formatting as RewardTrainer)."""
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=DTYPE).to(DEVICE).eval()
    scores = []
    for batch in batches([tok.apply_chat_template(full, tokenize=False) for _, full in convs], batch_size):
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        scores += model(**enc).logits[:, 0].float().tolist()
    del model
    return np.array(scores)


@torch.no_grad()
def completion_logps(path, convs, batch_size):
    """Sum of log-probabilities of the answer tokens, given the prompt."""
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE).to(DEVICE).eval()
    out = []
    for batch in batches(convs, batch_size):
        prompts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p, _ in batch]
        fulls = [tok.apply_chat_template(f, tokenize=False) for _, f in batch]
        n_prompt = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
        enc = tok(fulls, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        logp = torch.log_softmax(model(**enc).logits[:, :-1].float(), -1)
        tok_logp = logp.gather(-1, enc.input_ids[:, 1:, None])[..., 0]  # log p(token t | tokens < t)
        mask = enc.attention_mask[:, 1:].clone()
        for i, n in enumerate(n_prompt):
            mask[i, : n - 1] = 0  # only the answer's tokens count
        out += (tok_logp * mask).sum(-1).tolist()
    del model
    return np.array(out)


def ks(n):
    return sorted({k for k in (1, 2, 4, n) if k <= n})


def grade(scores, rows, rng, n_subsets=50):
    """All metrics for one scorer. scores: flat array, n per problem."""
    n = len(rows[0]["completions"])
    S = scores.reshape(len(rows), n)
    P = np.array([[k == r["total"] for k in r["passed"]] for r in rows])
    L = np.array([[len(c) for c in r["completions"]] for r in rows], dtype=float)

    wins = total = 0.0
    for s, p in zip(S, P):
        good, bad = s[p], s[~p]
        if len(good) and len(bad):
            diff = good[:, None] - bad[None, :]
            wins += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            total += diff.size
    res = {"pairwise acc": wins / max(total, 1), "pairs": int(total)}

    for k in ks(n):
        picked = []
        for s, p in zip(S, P):
            subsets = [np.arange(n)] if k == n else [rng.choice(n, k, replace=False) for _ in range(n_subsets)]
            picked.append(np.mean([p[sub[np.argmax(s[sub])]] for sub in subsets]))
        res[f"best-of-{k}"] = float(np.mean(picked))
        res[f"pass@{k}"] = float(np.mean([pass_at_k(n, c, k) for c in P.sum(1)]))

    # Within-problem correlations: subtract each problem's mean so "easy problem" isn't confused with "long answer".
    def corr(a, b):
        a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
    res["corr(score, length)"] = corr(S, L)
    res["corr(pass, length)"] = corr(P.astype(float), L)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=["outputs/eval/sft_kodcode_eval_n10_t0.8/samples.jsonl"])
    ap.add_argument("--rm", nargs="*", default=[], help="reward model checkpoints")
    ap.add_argument("--implicit", nargs=2, metavar=("DPO", "REF"), help="DPO checkpoint and its reference (SFT) model")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first N problems only (smoke tests)")
    ap.add_argument("--out", default="outputs/reward_eval.md")
    args = ap.parse_args()

    L = ["# Reward models vs. the unit tests", "",
         "Each scorer ranks the saved eval samples; the tests say which ones actually pass.",
         "*pairwise acc*: passing answer scored above a failing one for the same problem (0.5 = chance).",
         "*best-of-k*: pass rate of the top-scored of k samples, between pass@1 (random pick) and pass@k (perfect pick).",
         "*corr*: within-problem correlation with answer length; a good scorer tracks length only as much as passing does.", ""]
    for path in args.samples:
        rows = [json.loads(l) for l in open(path)][: args.limit]
        convs = conversations(rows)
        scorers = {Path(rm).name: rm_scores(rm, convs, args.batch_size) for rm in args.rm}
        if args.implicit:
            dpo, ref = args.implicit
            scorers[f"implicit ({Path(dpo).name})"] = (completion_logps(dpo, convs, args.batch_size)
                                                      - completion_logps(ref, convs, args.batch_size))
        scorers["random"] = np.random.default_rng(0).random(len(convs))

        n = len(rows[0]["completions"])
        L += [f"## Samples: `{path}`", "", f"{len(rows)} problems x {n} samples.", "",
              "| scorer | pairwise acc | " + " | ".join(f"best-of-{k}" for k in ks(n)) + " | corr(score, length) |",
              "|---|---|" + "---|" * len(ks(n)) + "---|"]
        for name, scores in scorers.items():
            r = grade(scores, rows, np.random.default_rng(0))
            L.append(f"| {name} | {r['pairwise acc']:.3f} | " + " | ".join(f"{r[f'best-of-{k}']:.3f}" for k in ks(n))
                     + f" | {r['corr(score, length)']:+.2f} |")
        L.append("| *perfect picker (pass@k)* | 1 | " + " | ".join(f"{r[f'pass@{k}']:.3f}" for k in ks(n))
                 + f" | corr(pass, length) = {r['corr(pass, length)']:+.2f} |")
        L += ["", f"{r['pairs']} (passing, failing) pairs within problems.", ""]

    report = "\n".join(L) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report)
    print(report)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
