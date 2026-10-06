"""An LLM as judge: score answers by asking a model whether the code is correct.

Two modes, to see what "thinking first" adds:
  direct  answer YES or NO immediately; score = P(YES) from the next-token probabilities (one forward pass)
  cot     reason step by step first (check the logic, trace the example test), then get the same YES/NO
          question as a follow-up; score = P(YES) after the reasoning, averaged over --votes reasonings.
          (Reading P(YES) instead of parsing a verdict line works for models that don't follow formats.)

The judge sees what the policy saw (problem + the one example test), never the hidden tests.
Writes a flat list of scores in samples.jsonl order; grade it with
scripts/eval_reward.py --scores NAME=PATH.

  python scripts/judge.py --model Qwen/Qwen2.5-Coder-7B-Instruct --mode cot \
      --samples outputs/eval/sft_kodcode_eval_n10_t0.8/samples.jsonl --out outputs/judge/coder7b_cot.json
"""

import argparse
import json
import math
from pathlib import Path

from codealign.data import extract_code

SYSTEM = "You are an expert Python reviewer. You judge whether code solves the task correctly."
ASK = {
    "direct": "Will this solution pass the problem's hidden unit tests? Answer with exactly one word: YES or NO.",
    "cot": ("Will this solution pass the problem's hidden unit tests? Think step by step: check the logic against "
            "the task, consider edge cases, and trace the example test. End with your conclusion."),
}
FOLLOW_UP = "So, will the solution pass the hidden unit tests? Answer with exactly one word: YES or NO."


def judge_messages(row, completion, mode):
    code = extract_code(completion).strip()
    user = f"Problem:\n{row['prompt']}\n\nCandidate solution:\n```python\n{code}\n```\n\n{ASK[mode]}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def p_yes(top_logprobs: dict[str, float]) -> float:
    """P(YES) / (P(YES) + P(NO)) from the first token's top alternatives."""
    yes = sum(math.exp(lp) for t, lp in top_logprobs.items() if t.strip().upper() == "YES")
    no = sum(math.exp(lp) for t, lp in top_logprobs.items() if t.strip().upper() == "NO")
    return yes / (yes + no) if yes + no > 0 else 0.5


def follow_ups(conv, reasonings):
    return [conv + [{"role": "assistant", "content": r}, {"role": "user", "content": FOLLOW_UP}] for r in reasonings]


def run_vllm(model, convs, mode, votes, max_tokens):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, seed=0, max_model_len=4096)
    yes_no = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    def ask(cs):
        return [p_yes({lp.decoded_token: lp.logprob for lp in o.outputs[0].logprobs[0].values()})
                for o in llm.chat(cs, yes_no)]

    if mode == "direct":
        return ask(convs), []
    outs = llm.chat(convs, SamplingParams(n=votes, temperature=0.7, max_tokens=max_tokens, seed=0))
    texts = [[c.text for c in o.outputs] for o in outs]
    p = ask([f for conv, t in zip(convs, texts) for f in follow_ups(conv, t)])
    return [sum(p[i * votes:(i + 1) * votes]) / votes for i in range(len(convs))], texts


def run_hf(model, convs, mode, votes, max_tokens):
    """Slow fallback for local smoke tests."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(model)
    encode = lambda conv: tok(tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True),
                              return_tensors="pt")

    @torch.no_grad()
    def ask(conv):
        top = torch.topk(torch.log_softmax(lm(**encode(conv)).logits[0, -1].float(), -1), 20)
        return p_yes({tok.decode(i): v.item() for i, v in zip(top.indices, top.values)})

    scores, texts = [], []
    for conv in convs:
        if mode == "direct":
            scores.append(ask(conv))
            continue
        ids = encode(conv)
        with torch.no_grad():
            out = lm.generate(**ids, do_sample=True, temperature=0.7, max_new_tokens=max_tokens,
                              num_return_sequences=votes, pad_token_id=tok.eos_token_id)
        t = [tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True) for o in out]
        scores.append(sum(ask(f) for f in follow_ups(conv, t)) / votes)
        texts.append(t)
    return scores, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["direct", "cot"], default="direct")
    ap.add_argument("--samples", default="outputs/eval/sft_kodcode_eval_n10_t0.8/samples.jsonl")
    ap.add_argument("--votes", type=int, default=4, help="cot: reasonings sampled per answer")
    ap.add_argument("--max-tokens", type=int, default=1024, help="cot: reasoning budget")
    ap.add_argument("--limit", type=int, help="first N problems only (smoke tests)")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples)][: args.limit]
    convs = [judge_messages(r, c, args.mode) for r in rows for c in r["completions"]]
    run = run_vllm if args.backend == "vllm" else run_hf
    scores, texts = run(args.model, convs, args.mode, args.votes, args.max_tokens)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "mode": args.mode, "samples": args.samples, "scores": scores}))
    if texts:  # a few reasoning traces, to read how the judge decides
        examples = [{"messages": c, "judgements": t} for c, t in zip(convs[:20], texts[:20])]
        out.with_suffix(".examples.json").write_text(json.dumps(examples, indent=1))
    print(f"{len(scores)} answers judged, mean score {sum(scores) / len(scores):.3f} -> {out}")


if __name__ == "__main__":
    main()
