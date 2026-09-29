#!/bin/bash
# Full pipeline on one GPU: baselines -> SFT -> DPO -> GRPO, evaluating after each stage.
# Every model is scored with the same eval (MBPP-sanitized test, greedy pass@1 + sampled pass@10).
#
# Usually submitted by slurm/launch.sh (which downloads models first). Directly:
#   sbatch slurm/pipeline.sbatch                # everything
#   STAGES="grpo" sbatch slurm/pipeline.sbatch  # just one stage (needs earlier outputs)
#   STAGES="sft_kodcode" SBATCH_TIMELIMIT=16:00:00 bash slurm/launch.sh   # large SFT (not in the default stages)
#   STAGES="dpo grpo" SFT_CKPT=checkpoints/sft_kodcode bash slurm/launch.sh # DPO/GRPO on top of it
#   STAGES="analyze" bash slurm/launch.sh      # beyond-pass@k report -> outputs/analysis.md
#
#SBATCH --job-name=codealign-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1        # needs bf16; older GPUs (T4, RTX 6000) lack it
#SBATCH --cpus-per-task=16          # the sandbox runs tests in parallel on CPUs
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail

# --- environment (TAMU Grace; edit for other clusters) ---
module purge
module load GCCcore/13.2.0 Python/3.11.5
module load CUDA/13.0.0 || echo "warning: CUDA module not loaded"   # nvcc, in case a kernel needs JIT compiling
source $SCRATCH/envs/codealign/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/.cache/huggingface}   # never $HOME: 10k-file quota
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # compute nodes have no internet: pre-download on the login node
export VLLM_CONFIG_ROOT=$SCRATCH/.config/vllm VLLM_CACHE_ROOT=$SCRATCH/.cache/vllm   # vLLM defaults to ~/.config (home quota)
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0   # its sampler JIT-compiles CUDA code; PyTorch's sampler is fine
export XDG_CACHE_HOME=$SCRATCH/.cache TRITON_CACHE_DIR=$SCRATCH/.cache/triton CUDA_CACHE_PATH=$SCRATCH/.cache/nv   # JIT caches, not home
set -u  # only after module/activate, whose scripts reference unset variables
REPORT_TO=${REPORT_TO:-none}        # set REPORT_TO=wandb to get training curves

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data checkpoints outputs
BASE=${BASE:-Qwen/Qwen2.5-0.5B}
STAGES=${STAGES:-"baseline sft dpo grpo"}
EVAL=outputs/eval

echo "=== checks: GPU and sandbox ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
pytest -q

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

if run sft_kodcode; then
    # SFT on MBPP train + ~480k KodCode problems (one epoch: several hours). Separate
    # checkpoint, so it shows up as its own row next to the MBPP-only SFT.
    echo "=== SFT (KodCode) ==="
    python train/sft.py --model "$BASE" --data kodcode --epochs 1 --batch-size 16 \
        --out checkpoints/sft_kodcode --report-to "$REPORT_TO"
    evaluate checkpoints/sft_kodcode
fi

# DPO and GRPO start from SFT_CKPT. Outputs are suffixed by its name, so e.g.
# SFT_CKPT=checkpoints/sft_kodcode writes checkpoints/dpo_kodcode, grpo_kodcode.
SFT_CKPT=${SFT_CKPT:-checkpoints/sft}
SFT_NAME=$(basename "$SFT_CKPT")
SUFFIX=${SFT_NAME#sft}                          # "" for sft, "_kodcode" for sft_kodcode
SAMPLES=$EVAL/${SFT_NAME}_train_n8_t1.0/samples.jsonl   # evaluate.py names runs after the model dir
PAIRS=data/dpo_pairs$SUFFIX.jsonl

if run dpo || run grpo; then
    # Sample 8 solutions per train problem from the SFT model. Used for DPO pairs
    # and for GRPO's difficulty filter.
    if [[ ! -f $SAMPLES ]]; then
        python scripts/evaluate.py --model "$SFT_CKPT" --chat --split train --n 8 --temperature 1.0
    fi
fi

if run dpo; then
    echo "=== DPO (from $SFT_CKPT) ==="
    python scripts/make_dpo_pairs.py "$SAMPLES" --out "$PAIRS"
    python train/dpo.py --model "$SFT_CKPT" --pairs "$PAIRS" --out checkpoints/dpo$SUFFIX --report-to "$REPORT_TO"
    evaluate checkpoints/dpo$SUFFIX
fi

if run grpo; then
    echo "=== GRPO (from $SFT_CKPT) ==="
    python train/grpo.py --model "$SFT_CKPT" --out checkpoints/grpo$SUFFIX --vllm \
        --filter-samples "$SAMPLES" --report-to "$REPORT_TO"
    evaluate checkpoints/grpo$SUFFIX
fi

if run analyze; then
    # Beyond pass@k: new ability vs. sharpening, reward hacking, hallucination, diversity.
    # CPU only (re-runs the saved eval samples), but it executes model code, so it runs in a job.
    echo "=== analysis ==="
    python scripts/analyze.py --eval-dir $EVAL --out outputs/analysis.md
fi

python scripts/summarize.py $EVAL
