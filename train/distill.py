"""On-policy distillation with TRL's DistillationTrainer.

The student writes its own answers to the prompts; then, token by token, it is
pulled toward the teacher's next-token distribution on those answers. No tests
and no reward: the teacher's probabilities are the whole signal. Compare with
GRPO from the same student (reward = tests passed, one number per answer) and
with plain SFT on the teacher's answers (scripts/make_distill_data.py).

Student and teacher must share a vocabulary (any two Qwen2.5 sizes do).

  python train/distill.py --model checkpoints/sft_kodcode --teacher checkpoints/grpo_kodcode_1.5b \
      --out checkpoints/kd_onpolicy_kodcode --vllm
"""

import argparse

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import DistillationConfig, DistillationTrainer

from codealign.data import load_kodcode_problems, load_split


def build_dataset(n_kodcode: int) -> Dataset:
    problems = load_split("train") + load_kodcode_problems(n_kodcode)
    return Dataset.from_list([{"prompt": p.messages()} for p in problems])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/sft_kodcode", help="student")
    ap.add_argument("--teacher", default="checkpoints/grpo_kodcode_1.5b")
    ap.add_argument("--out", default="checkpoints/kd_onpolicy_kodcode")
    ap.add_argument("--n-kodcode", type=int, default=20000, help="KodCode prompts added to MBPP train's")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="divergence: 0 = forward KL (cover all teacher modes), 1 = reverse KL (mode-seeking), 0.5 = JSD")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling and loss temperature")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--max-steps", type=int, default=400, help="400 x 32 = 12.8k student answers")
    ap.add_argument("--vllm", action="store_true", help="student generates with vLLM (colocated on the training GPU)")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--report-to", default="none")
    args = ap.parse_args()

    dtype = "bfloat16" if torch.cuda.is_available() else "float32"  # the teacher kwargs need a string
    config = DistillationConfig(
        output_dir=args.out,
        beta=args.beta,
        temperature=args.temperature,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.lr * (10 if args.lora else 1),
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=0.05,
        bf16=torch.cuda.is_available(),
        model_init_kwargs={"dtype": dtype},
        teacher_model_init_kwargs={"dtype": dtype},
        use_vllm=args.vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=2,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        report_to=args.report_to,
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM") if args.lora else None

    trainer = DistillationTrainer(
        model=args.model,
        teacher_model=args.teacher,
        args=config,
        train_dataset=build_dataset(args.n_kodcode),
        peft_config=peft_config,
    )
    trainer.train()

    model = trainer.model.merge_and_unload() if args.lora else trainer.model
    model.save_pretrained(args.out)
    trainer.processing_class.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
