"""Step 4: GRPO with unit-test execution as the reward, using TRL's GRPOTrainer.

For each prompt the model samples a group of G completions. Each one is run
against the tests; its advantage is (reward - group mean) / group std. So a
problem the model always solves (or never solves) produces zero advantage
and zero gradient: use --filter-samples to train only on problems with a
mixed pass rate.

  python train/grpo.py --model checkpoints/sft --out checkpoints/grpo --vllm \
      --filter-samples outputs/eval/sft_train_n8_t1.0/samples.jsonl

RLHF: --reward rm:PATH rewards answers with a learned reward model (train/reward.py)
instead of the tests. The tests still run on every answer, with weight 0: they are
logged as rewards/test_gold/mean (the true pass rate) but never trained on. With
--log-kl the KL from the starting model is logged too, so proxy reward, gold reward
and KL can be plotted against each other (scripts/overoptimization.py).

  python train/grpo.py --model checkpoints/sft_kodcode --reward rm:checkpoints/rm_kodcode --beta 0.02 \
      --log-kl --epochs 15 --out checkpoints/rlhf_rm0.02_kodcode --vllm --filter-samples ...
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from codealign.data import extract_code, load_split
from codealign.sandbox import run_many


def make_reward(kind: str, name: str | None = None):
    def test_reward(completions, tests, setup, **kwargs) -> list[float]:
        # Conversational datasets give completions as [{"role": "assistant", "content": ...}].
        codes = [extract_code(c[0]["content"]) for c in completions]
        results = run_many([{"program": c, "tests": t, "setup": s} for c, t, s in zip(codes, tests, setup)])
        if kind == "binary":
            return [float(r.all_passed) for r in results]
        return [r.frac for r in results]

    test_reward.__name__ = name or f"test_{kind}"
    return test_reward


def build_dataset(filter_samples: str | None) -> Dataset:
    problems = load_split("train")
    if filter_samples:
        keep = set()
        for line in open(filter_samples):
            row = json.loads(line)
            n_pass = sum(k == row["total"] for k in row["passed"])
            if 0 < n_pass < len(row["passed"]):
                keep.add(row["task_id"])
        print(f"difficulty filter: keeping {len(keep)}/{len(problems)} problems with mixed pass rate")
        problems = [p for p in problems if p.task_id in keep]
    return Dataset.from_list([{"prompt": p.messages(), "tests": p.tests, "setup": p.setup} for p in problems])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/sft")
    ap.add_argument("--out", default="checkpoints/grpo")
    ap.add_argument("--reward", default="frac",
                    help="frac = fraction of tests passed (denser signal); binary = all tests pass; "
                         "rm:PATH = learned reward model (RLHF), with the tests logged but not trained on")
    ap.add_argument("--log-kl", action="store_true",
                    help="log the KL from the starting model even at --beta 0 (uses beta=1e-6: same objective in practice)")
    ap.add_argument("--filter-samples", help="samples.jsonl from evaluate.py --split train --n 8")
    ap.add_argument("--num-generations", type=int, default=8, help="group size G")
    ap.add_argument("--batch-size", type=int, default=16, help="completions per device per step")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=3e-6)  # 1e-6 for 70 steps left the reward flat
    ap.add_argument("--temperature", type=float, default=0.8, help="1.0 drifted into garbage tokens")
    ap.add_argument("--beta", type=float, default=0.0, help="KL penalty to the reference model")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--vllm", action="store_true", help="generate with vLLM (colocated on the training GPU)")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--report-to", default="none")
    args = ap.parse_args()

    if args.reward.startswith("rm:"):
        # Weight 0: the tests are only measured (gold reward), the reward model is what's optimized (proxy).
        reward_funcs, weights = [args.reward[3:], make_reward("binary", "test_gold")], [1.0, 0.0]
    elif args.log_kl and args.reward != "binary":
        # Same gold as the RLHF runs (all tests pass), logged next to the reward trained on.
        reward_funcs, weights = [make_reward(args.reward), make_reward("binary", "test_gold")], [1.0, 0.0]
    else:
        reward_funcs, weights = [make_reward(args.reward)], None
    # TRL skips the reference model, and with it the KL, when beta == 0.
    beta = args.beta if args.beta or not args.log_kl else 1e-6

    config = GRPOConfig(
        output_dir=args.out,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        mask_truncated_completions=True,  # completions cut off at max length are excluded from the loss
        beta=beta,
        reward_weights=weights,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr * (10 if args.lora else 1),
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=0.05,
        bf16=torch.cuda.is_available(),
        model_init_kwargs={"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
        use_vllm=args.vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=2,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to=args.report_to,
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM") if args.lora else None

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=build_dataset(args.filter_samples),
        peft_config=peft_config,
    )
    trainer.train()
    # Per-step rewards and KL, for the over-optimization curves.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))

    model = trainer.model.merge_and_unload() if args.lora else trainer.model
    model.save_pretrained(args.out)
    trainer.processing_class.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
