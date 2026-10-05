"""Step 3a (RLHF): train a reward model on (passing, failing) answer pairs with TRL's RewardTrainer.

The reward model is the language model with its next-token layer replaced by a
single output, the score. It is trained with the Bradley-Terry loss
-log sigmoid(score(chosen) - score(rejected)): passing answers should score
higher than failing ones on the same problem. Only score differences matter.

Pairs come from scripts/make_dpo_pairs.py. 5% of problems are held out to
report pairwise accuracy during training; scripts/eval_reward.py then tests the
model on MBPP eval samples it never saw.

  python train/reward.py --model checkpoints/sft_kodcode --pairs data/rm_pairs_kodcode.jsonl \
      --out checkpoints/rm_kodcode
"""

import argparse
import json
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import RewardConfig, RewardTrainer


def load_pairs(path: str, holdout: float, seed: int = 0) -> tuple[Dataset, Dataset]:
    """Split by problem, not by pair: pairs from one problem share a prompt."""
    rows = [json.loads(l) for l in open(path)]
    tasks = sorted({r["task_id"] for r in rows})
    random.Random(seed).shuffle(tasks)
    held = set(tasks[: max(1, int(len(tasks) * holdout))])
    cols = ("prompt", "chosen", "rejected")
    train = [{k: r[k] for k in cols} for r in rows if r["task_id"] not in held]
    test = [{k: r[k] for k in cols} for r in rows if r["task_id"] in held]
    print(f"{len(rows)} pairs from {len(tasks)} problems: {len(train)} train, {len(test)} held out ({len(held)} problems)")
    return Dataset.from_list(train).shuffle(seed=seed), Dataset.from_list(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/sft_kodcode", help="start from this language model")
    ap.add_argument("--pairs", default="data/rm_pairs_kodcode.jsonl")
    ap.add_argument("--out", default="checkpoints/rm_kodcode")
    ap.add_argument("--epochs", type=float, default=1, help="reward models overfit quickly; 1 epoch is standard")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8, help="pairs per device per step")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--report-to", default="none")
    args = ap.parse_args()

    train, test = load_pairs(args.pairs, args.holdout)
    config = RewardConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr * (10 if args.lora else 1),
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=1024,  # longer pairs are dropped, not truncated
        center_rewards_coefficient=0.01,  # keep scores near 0, so they can be used as an RL reward directly
        bf16=torch.cuda.is_available(),
        model_init_kwargs={"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=0.1,  # 10 evaluations over the run
        save_strategy="no",
        report_to=args.report_to,
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="SEQ_CLS") if args.lora else None

    trainer = RewardTrainer(model=args.model, args=config, train_dataset=train, eval_dataset=test, peft_config=peft_config)
    trainer.train()
    print("held-out pairs:", trainer.evaluate())

    model = trainer.model.merge_and_unload() if args.lora else trainer.model
    model.save_pretrained(args.out)
    trainer.processing_class.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
