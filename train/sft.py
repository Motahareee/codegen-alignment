"""Step 1: supervised fine-tuning on gold MBPP solutions with TRL's SFTTrainer.

The model sees (prompt -> ```python gold code```) pairs. Loss is computed only
on the completion tokens (TRL's default for prompt/completion datasets).

  python train/sft.py --model Qwen/Qwen2.5-0.5B --out checkpoints/sft
  python train/sft.py --data kodcode --epochs 1 --out checkpoints/sft_kodcode   # + 480k KodCode problems
"""

import argparse

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from codealign.data import load_kodcode_sft, load_split


def build_dataset(data: str, max_examples: int | None) -> Dataset:
    rows = []
    for p in load_split("train"):
        answer = f"```python\n{p.reference.strip()}\n```"
        rows.append({"prompt": p.messages(), "completion": [{"role": "assistant", "content": answer}]})
    ds = Dataset.from_list(rows)
    if data == "kodcode":
        ds = concatenate_datasets([ds, load_kodcode_sft(max_examples)]).shuffle(seed=0)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--data", default="mbpp", choices=["mbpp", "kodcode"], help="kodcode = MBPP train + KodCode-V1")
    ap.add_argument("--max-examples", type=int, help="cap on KodCode examples (default: all)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs (smoke tests)")
    ap.add_argument("--lora", action="store_true", help="train LoRA adapters instead of all weights")
    ap.add_argument("--report-to", default="none", help="e.g. wandb")
    args = ap.parse_args()

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr * (10 if args.lora else 1),  # LoRA wants a higher LR
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # float < 1 = fraction of total steps
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=1024,
        eos_token="<|im_end|>",  # Qwen base models use <|endoftext|>; chat turns end with <|im_end|>
        bf16=torch.cuda.is_available(),
        model_init_kwargs={"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
        logging_steps=5,
        save_strategy="steps" if args.data == "kodcode" else "no",  # long run: keep a checkpoint in case of timeout
        save_steps=2000,
        save_total_limit=1,
        dataset_num_proc=8,
        report_to=args.report_to,
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM") if args.lora else None

    trainer = SFTTrainer(model=args.model, args=config, train_dataset=build_dataset(args.data, args.max_examples), peft_config=peft_config)
    trainer.train()

    # Save a plain merged model so evaluate.py / vLLM / later stages can load it directly.
    model = trainer.model.merge_and_unload() if args.lora else trainer.model
    tok = trainer.processing_class
    model.generation_config.eos_token_id = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
