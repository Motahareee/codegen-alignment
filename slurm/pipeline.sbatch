#!/bin/bash
# Full pipeline on one GPU: baselines -> SFT -> DPO -> GRPO, evaluating after each stage.
# Every model is scored with the same eval (MBPP-sanitized test, greedy pass@1 + sampled pass@10).
#
#   sbatch slurm/pipeline.sbatch                # everything
#   STAGES="grpo" sbatch slurm/pipeline.sbatch  # just one stage (needs earlier outputs)
#
#SBATCH --job-name=codealign-pipeline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16          # the sandbox runs tests in parallel on CPUs
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# --- environment: edit for your cluster ---
# module load cuda/12.4
source /opt/venv/bin/activate
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
REPORT_TO=${REPORT_TO:-none}        # set REPORT_TO=wandb to get training curves

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data checkpoints outputs
BASE=${BASE:-Qwen/Qwen2.5-0.5B}
STAGES=${STAGES:-"baseline sft dpo grpo"}
EVAL=outputs/eval

evaluate() {  # $1 = model path/id
    python scripts/evaluate.py --model "$1" --chat                            # greedy pass@1
    python scripts/evaluate.py --model "$1" --chat --n 10 --temperature 0.8   # pass@1 / pass@10 sampled
}

run() { [[ " $STAGES " == *" $1 "* ]]; }

if run baseline; then
    echo "=== baselines ==="
    evaluate "$BASE"
    evaluate Qwen/Qwen2.5-0.5B-Instruct   # reference point: a model post-trained by Qwen
fi

if run sft; then
    echo "=== SFT ==="
    python train/sft.py --model "$BASE" --out checkpoints/sft --report-to "$REPORT_TO"
    evaluate checkpoints/sft
fi

if run dpo || run grpo; then
    # Sample 8 solutions per train problem from the SFT model. Used for DPO pairs
    # and for GRPO's difficulty filter.
    if [[ ! -f $EVAL/sft_train_n8_t1.0/samples.jsonl ]]; then
        python scripts/evaluate.py --model checkpoints/sft --chat --split train --n 8 --temperature 1.0
    fi
fi

if run dpo; then
    echo "=== DPO ==="
    python scripts/make_dpo_pairs.py $EVAL/sft_train_n8_t1.0/samples.jsonl --out data/dpo_pairs.jsonl
    python train/dpo.py --model checkpoints/sft --pairs data/dpo_pairs.jsonl --out checkpoints/dpo --report-to "$REPORT_TO"
    evaluate checkpoints/dpo
fi

if run grpo; then
    echo "=== GRPO ==="
    python train/grpo.py --model checkpoints/sft --out checkpoints/grpo --vllm \
        --filter-samples $EVAL/sft_train_n8_t1.0/samples.jsonl --report-to "$REPORT_TO"
    evaluate checkpoints/grpo
fi

python scripts/summarize.py $EVAL
