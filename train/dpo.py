"""Step 2: DPO on self-generated (passing, failing) pairs with TRL's DPOTrainer.

The reference model is a frozen copy of the starting model (the SFT
checkpoint). With --lora, TRL uses the base weights with adapters disabled
as the reference, so no second copy is needed in memory.

  python train/dpo.py --model checkpoints/sft --pairs data/dpo_pairs.jsonl --out checkpoints/dpo
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/sft")
    ap.add_argument("--pairs", default="data/dpo_pairs.jsonl")
    ap.add_argument("--out", default="checkpoints/dpo")
    ap.add_argument("--beta", type=float, default=0.1, help="strength of the pull back toward the reference model")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--report-to", default="none")
    args = ap.parse_args()

    dataset = load_dataset("json", data_files=args.pairs, split="train").remove_columns("task_id")

    config = DPOConfig(
        output_dir=args.out,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr * (10 if args.lora else 1),
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=1024,
        bf16=torch.cuda.is_available(),
        model_init_kwargs={"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
        logging_steps=5,
        save_strategy="no",
        report_to=args.report_to,
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM") if args.lora else None

    trainer = DPOTrainer(model=args.model, args=config, train_dataset=dataset, peft_config=peft_config)
    trainer.train()

    model = trainer.model.merge_and_unload() if args.lora else trainer.model
    model.save_pretrained(args.out)
    trainer.processing_class.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
