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
# A bigger base model: its checkpoints get a size tag (sft_kodcode_1.5b, grpo_kodcode_1.5b, ...)
#   BASE=Qwen/Qwen2.5-1.5B STAGES="baseline sft_kodcode grpo" SFT_CKPT=checkpoints/sft_kodcode_1.5b \
#     SFT_ARGS="--batch-size 8 --grad-accum 4" SBATCH_TIMELIMIT=2-00:00:00 bash slurm/launch.sh
# Distillation into the 0.5B KodCode student (defaults: TEACHER=checkpoints/grpo_kodcode_1.5b,
# STUDENT=checkpoints/sft_kodcode):
#   STAGES="kd_seq kd_onpolicy" bash slurm/launch.sh
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
SIZE=$(basename "$BASE"); SIZE=${SIZE##*-}      # Qwen/Qwen2.5-1.5B -> 1.5B
TAG=$([[ $SIZE == 0.5B ]] && echo "" || echo "_${SIZE,,}")   # "" for 0.5B, "_1.5b" for 1.5B
STAGES=${STAGES:-"baseline sft dpo grpo"}
EVAL=outputs/eval

# Each stage prints a header; when the next one starts, the previous one's duration is printed.
stage() {
    if [[ -n ${STAGE_NAME:-} ]]; then
        local t=$((SECONDS - STAGE_START))
        printf -- "--- %s took %dh%02dm\n" "$STAGE_NAME" $((t / 3600)) $((t % 3600 / 60))
    fi
    STAGE_NAME=$1 STAGE_START=$SECONDS
    [[ $1 == done ]] || echo "=== $1 === ($(date '+%F %H:%M'))"
}

stage "checks: GPU and sandbox"
nvidia-smi --query-gpu=name,memory.total --format=csv
pytest -q

evaluate() {  # $1 = model path/id
    python scripts/evaluate.py --model "$1" --chat                            # greedy pass@1
    python scripts/evaluate.py --model "$1" --chat --n 10 --temperature 0.8   # pass@1 / pass@10 sampled
}

run() { [[ " $STAGES " == *" $1 "* ]]; }

if run baseline; then
    stage "baselines"
    evaluate "$BASE"
    evaluate "$BASE-Instruct"   # reference point: a model post-trained by Qwen
fi

if run sft; then
    stage "SFT"
    python train/sft.py --model "$BASE" --out checkpoints/sft$TAG --report-to "$REPORT_TO"
    evaluate checkpoints/sft$TAG
fi

if run sft_kodcode; then
    # SFT on MBPP train + ~480k KodCode problems (one epoch: several hours). Separate
    # checkpoint, so it shows up as its own row next to the MBPP-only SFT.
    # SFT_ARGS: extra flags, e.g. a smaller batch for bigger models; KODCODE_MAX caps the examples.
    stage "SFT (KodCode, $BASE)"
    python train/sft.py --model "$BASE" --data kodcode --epochs 1 ${SFT_ARGS:---batch-size 16} \
        ${KODCODE_MAX:+--max-examples $KODCODE_MAX} --out checkpoints/sft_kodcode$TAG --report-to "$REPORT_TO"
    evaluate checkpoints/sft_kodcode$TAG
fi

# DPO and GRPO start from SFT_CKPT. Outputs are suffixed by its name, so e.g.
# SFT_CKPT=checkpoints/sft_kodcode writes checkpoints/dpo_kodcode, grpo_kodcode.
SFT_CKPT=${SFT_CKPT:-checkpoints/sft$TAG}
SFT_NAME=$(basename "$SFT_CKPT")
SUFFIX=${SFT_NAME#sft}                          # "" for sft, "_kodcode" for sft_kodcode, "_kodcode_1.5b" ...
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
    stage "DPO (from $SFT_CKPT)"
    python scripts/make_dpo_pairs.py "$SAMPLES" --out "$PAIRS"
    python train/dpo.py --model "$SFT_CKPT" --pairs "$PAIRS" --out checkpoints/dpo$SUFFIX --report-to "$REPORT_TO"
    evaluate checkpoints/dpo$SUFFIX
fi

if run grpo; then
    stage "GRPO (from $SFT_CKPT)"
    python train/grpo.py --model "$SFT_CKPT" --out checkpoints/grpo$SUFFIX --vllm \
        --filter-samples "$SAMPLES" --report-to "$REPORT_TO"
    evaluate checkpoints/grpo$SUFFIX
fi

# Distillation: a bigger TEACHER into a small STUDENT, compared with GRPO from the same student.
#   kd_seq:      SFT on the teacher's own answers to KodCode problems that pass the tests
#   kd_onpolicy: the student writes answers; the teacher's token probabilities on them are the target
TEACHER=${TEACHER:-checkpoints/grpo_kodcode_1.5b}
STUDENT=${STUDENT:-checkpoints/sft_kodcode}
STUDENT_NAME=$(basename "$STUDENT")
KD_SUFFIX=${STUDENT_NAME#sft}                   # sft_kodcode -> kd_seq_kodcode, kd_onpolicy_kodcode
KD_PROBLEMS=${KD_PROBLEMS:-20000}

if run kd_seq; then
    stage "distillation: SFT on $TEACHER's answers -> $STUDENT"
    TEACHER_SAMPLES=$EVAL/$(basename "$TEACHER")_kodcode_n4_t0.7/samples.jsonl
    if [[ ! -f $TEACHER_SAMPLES ]]; then
        python scripts/evaluate.py --model "$TEACHER" --chat --split kodcode --limit "$KD_PROBLEMS" --n 4 --temperature 0.7
    fi
    KD_DATA=data/kd_seq_$(basename "$TEACHER").jsonl
    python scripts/make_distill_data.py "$TEACHER_SAMPLES" --out "$KD_DATA"
    python train/sft.py --model "$STUDENT" --data "$KD_DATA" --epochs 1 --out checkpoints/kd_seq$KD_SUFFIX --report-to "$REPORT_TO"
    evaluate checkpoints/kd_seq$KD_SUFFIX
fi

if run kd_onpolicy; then
    stage "distillation: on-policy, $TEACHER -> $STUDENT"
    python train/distill.py --model "$STUDENT" --teacher "$TEACHER" --n-kodcode "$KD_PROBLEMS" \
        --out checkpoints/kd_onpolicy$KD_SUFFIX --vllm --report-to "$REPORT_TO"
    evaluate checkpoints/kd_onpolicy$KD_SUFFIX
fi

if run analyze; then
    # Beyond pass@k: new ability vs. sharpening, reward hacking, hallucination, diversity.
    # CPU only (re-runs the saved eval samples), but it executes model code, so it runs in a job.
    stage "analysis"
    python scripts/analyze.py --eval-dir $EVAL --out outputs/analysis.md
fi

python scripts/summarize.py $EVAL
stage done
