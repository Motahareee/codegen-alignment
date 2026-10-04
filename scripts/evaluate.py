"""Sample completions from a model, run them against tests, report pass@k.

Every method in this project (base, SFT, DPO, PPO, GRPO) is scored with this
same script, so results are directly comparable.

Examples:
  # instruct / chat model, greedy pass@1
  python scripts/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --chat

  # base model, 10 samples per problem for pass@1 and pass@10
  python scripts/evaluate.py --model Qwen/Qwen2.5-0.5B --n 10 --temperature 0.8

  # teacher answers on 20k KodCode problems, checked against their tests (distillation data)
  python scripts/evaluate.py --model checkpoints/grpo_kodcode_1.5b --chat --split kodcode --limit 20000 --n 4 --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from codealign.data import extract_code, load_kodcode_problems, load_split
from codealign.metrics import pass_at_k
from codealign.sandbox import run_many

# Base (non-chat) models get the prompt as plain text with the code fence
# already opened, so they continue straight into code.
BASE_TEMPLATE = "{prompt}\n\n```python\n"


def generate_vllm(model, prompts, chat, n, temperature, max_tokens, seed):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, seed=seed)
    params = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens,
                            stop=None if chat else ["```"], seed=seed)
    outputs = llm.chat(prompts, params) if chat else llm.generate(prompts, params)
    return [[o.text for o in out.outputs] for out in outputs]


def generate_hf(model, prompts, chat, n, temperature, max_tokens, seed):
    """Slow fallback for local smoke tests without vLLM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(model)
    results = []
    for p in prompts:
        text = tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) if chat else p
        ids = tok(text, return_tensors="pt")
        out = lm.generate(**ids, max_new_tokens=max_tokens, num_return_sequences=n,
                          do_sample=temperature > 0, temperature=temperature or None,
                          top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
        results.append([tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True) for o in out])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="eval", choices=["train", "eval", "kodcode"],
                    help="kodcode = KodCode problems with converted tests (use --limit; not an eval set)")
    ap.add_argument("--chat", action="store_true", help="use the tokenizer's chat template")
    ap.add_argument("--n", type=int, default=1, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, help="only the first N problems (smoke tests)")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    if args.split == "kodcode":
        problems = load_kodcode_problems(args.limit)
    else:
        problems = load_split(args.split)[: args.limit]
    prompts = [p.messages() if args.chat else BASE_TEMPLATE.format(prompt=p.prompt) for p in problems]

    t0 = time.time()
    generate = generate_vllm if args.backend == "vllm" else generate_hf
    completions = generate(args.model, prompts, args.chat, args.n, args.temperature, args.max_tokens, args.seed)
    t_gen = time.time() - t0

    # Base-model completions are already bare code (we opened the fence and stop at ```).
    codes = [[extract_code(c) if args.chat else c for c in cs] for cs in completions]
    jobs = [{"program": code, "tests": p.tests, "setup": p.setup}
            for p, cs in zip(problems, codes) for code in cs]
    t0 = time.time()
    flat = run_many(jobs)
    t_exec = time.time() - t0
    results = [flat[i * args.n:(i + 1) * args.n] for i in range(len(problems))]

    correct = [sum(r.all_passed for r in rs) for rs in results]
    ks = [k for k in (1, 5, 10, 50, 100) if k <= args.n]
    metrics = {f"pass@{k}": float(np.mean([pass_at_k(args.n, c, k) for c in correct])) for k in ks}
    metrics["mean_test_frac"] = float(np.mean([r.frac for rs in results for r in rs]))
    metrics["status"] = dict(Counter(r.status for r in flat))

    run_name = f"{Path(args.model).name}_{args.split}_n{args.n}_t{args.temperature}"
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "samples.jsonl", "w") as f:
        for p, cs, rs in zip(problems, completions, results):
            f.write(json.dumps({
                "task_id": p.task_id,
                "prompt": p.prompt,
                "completions": cs,
                "passed": [r.passed for r in rs],
                "total": rs[0].total,
                "status": [r.status for r in rs],
            }) + "\n")
    summary = {"model": args.model, "split": args.split, "n_problems": len(problems),
               **vars(args), **metrics, "gen_seconds": t_gen, "exec_seconds": t_exec}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"generation {t_gen:.0f}s, execution {t_exec:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
