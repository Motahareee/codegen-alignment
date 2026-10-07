#!/bin/bash
# Full pipeline on one GPU: baselines -> SFT -> DPO -> GRPO, evaluating after each stage.
# Every model is scored with the same eval (MBPP-sanitized test, greedy pass@1 + sampled pass@10).
#
# Usually submitted by slurm/launch.sh (which downloads models first). Directly:
#   sbatch slurm/pipeline.sbatch                # everything
#   STAGES="grpo" sbatch slurm/pipeline.sbatch  # just one stage (needs earlier outputs)
#   STAGES="sft_kodcode" SBATCH_TIMELIMIT=16:00:00 bash slurm/launch.sh   # KodCode SFT (not in the default stages)
#   STAGES="dpo grpo" SFT_CKPT=checkpoints/sft_kodcode bash slurm/launch.sh # DPO/GRPO on top of it
#   STAGES="analyze" bash slurm/launch.sh      # beyond-pass@k report -> outputs/analysis.md
#
# A bigger base model: its checkpoints get a size tag (sft_kodcode_1.5b, grpo_kodcode_1.5b, ...)
#   BASE=Qwen/Qwen2.5-1.5B STAGES="baseline sft_kodcode grpo" SFT_CKPT=checkpoints/sft_kodcode_1.5b \
#     SFT_ARGS="--batch-size 8 --grad-accum 4" GRPO_ARGS="--batch-size 8 --grad-accum 8" \
#     SBATCH_TIMELIMIT=3-00:00:00 bash slurm/launch.sh
#   (on a 40 GB A100: same effective batch sizes as the 0.5B runs, half the memory per step;
#    the 0.5B KodCode SFT took 13.7 h, so expect ~40 h for 1.5B)
# Distillation into the 0.5B KodCode student (defaults: TEACHER=checkpoints/grpo_kodcode_1.5b,
# STUDENT=checkpoints/sft_kodcode):
#   STAGES="kd_seq kd_onpolicy" bash slurm/launch.sh
# Longer on-policy runs: KD_NAME names the run, KD_ARGS go to train/distill.py, e.g.
#   STAGES="kd_onpolicy" KD_NAME=kd_onpolicy_fkl2k KD_ARGS="--beta 0 --max-steps 2000 --lr 1e-5 --n-kodcode 60000" \
#     SBATCH_TIMELIMIT=2-00:00:00 bash slurm/launch.sh
# Reward model (RLHF step 1) on RM_POLICY's own answers, graded against the tests -> outputs/reward_eval.md:
#   STAGES="reward" bash slurm/launch.sh
# ... starting from a bigger model (RM_INIT; answers and pairs still come from RM_POLICY):
#   EXTRA_MODELS=Qwen/Qwen2.5-Coder-7B-Instruct STAGES="reward" RM_INIT=Qwen/Qwen2.5-Coder-7B-Instruct \
#     RM_OUT=checkpoints/rm_kodcode_coder7b RM_ARGS="--lora --batch-size 4 --grad-accum 8" bash slurm/launch.sh
# LLM judges (direct answer vs reasoning first), graded next to the reward models -> outputs/reward_eval_judges.md:
#   EXTRA_MODELS=Qwen/Qwen2.5-Coder-7B-Instruct STAGES="judge" bash slurm/launch.sh
# RLHF: GRPO against that reward model at several KL penalties, plus a control rewarded by the tests,
# all logging the true pass rate -> outputs/overoptimization_kodcode.md:
#   STAGES="rlhf" SBATCH_TIMELIMIT=12:00:00 bash slurm/launch.sh
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
        --filter-samples "$SAMPLES" ${GRPO_ARGS:-} --report-to "$REPORT_TO"
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
    KD_OUT=checkpoints/${KD_NAME:-kd_onpolicy}$KD_SUFFIX
    stage "distillation: on-policy, $TEACHER -> $STUDENT ($KD_OUT ${KD_ARGS:-})"
    python train/distill.py --model "$STUDENT" --teacher "$TEACHER" --n-kodcode "$KD_PROBLEMS" \
        --out "$KD_OUT" --vllm ${KD_ARGS:-} --report-to "$REPORT_TO"
    evaluate "$KD_OUT"
fi

# Reward model: pairs of (passing, failing) answers from RM_POLICY on MBPP train + RM_KODCODE
# KodCode problems, 8 samples each. Graded on RM_POLICY's eval samples, next to DPO's implicit reward.
RM_POLICY=${RM_POLICY:-checkpoints/sft_kodcode}
RM_NAME=$(basename "$RM_POLICY")
RM_SUFFIX=${RM_NAME#sft}                        # sft_kodcode -> rm_kodcode
RM_KODCODE=${RM_KODCODE:-6000}                  # changing it? delete the old *_kodcode_n8_t1.0 samples first
RM_INIT=${RM_INIT:-$RM_POLICY}                  # model the reward model starts from
RM_OUT=${RM_OUT:-checkpoints/rm$RM_SUFFIX}
finished_rms() {  # reward models that finished saving (config.json is written last)
    for d in checkpoints/rm$RM_SUFFIX*; do [[ -f $d/config.json ]] && echo "$d"; done
}

if run reward; then
    stage "reward model (answers from $RM_POLICY)"
    RM_SAMPLES=($EVAL/${RM_NAME}_train_n8_t1.0/samples.jsonl $EVAL/${RM_NAME}_kodcode_n8_t1.0/samples.jsonl)
    [[ -f ${RM_SAMPLES[0]} ]] || python scripts/evaluate.py --model "$RM_POLICY" --chat --split train --n 8 --temperature 1.0
    [[ -f ${RM_SAMPLES[1]} ]] || python scripts/evaluate.py --model "$RM_POLICY" --chat --split kodcode \
        --limit "$RM_KODCODE" --n 8 --temperature 1.0
    python scripts/make_dpo_pairs.py "${RM_SAMPLES[@]}" --out data/rm_pairs$RM_SUFFIX.jsonl
    python train/reward.py --model "$RM_INIT" --pairs data/rm_pairs$RM_SUFFIX.jsonl \
        --out "$RM_OUT" ${RM_ARGS:-} --report-to "$REPORT_TO"
    IMPLICIT=$([[ -d checkpoints/dpo$RM_SUFFIX ]] && echo "--implicit checkpoints/dpo$RM_SUFFIX $RM_POLICY")
    python scripts/eval_reward.py --rm $(finished_rms) $IMPLICIT \
        --samples $EVAL/${RM_NAME}_eval_n10_t0.8/samples.jsonl --out outputs/reward_eval$RM_SUFFIX.md
fi

# LLM judges on RM_POLICY's eval answers: each model answers directly (P(YES)) and after reasoning
# (4 sampled verdicts). Graded with the reward models on the same exam.
JUDGE_MODELS=${JUDGE_MODELS:-"Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct"}

if run judge; then
    JUDGE_SAMPLES=$EVAL/${RM_NAME}_eval_n10_t0.8/samples.jsonl
    JUDGE_SCORES=()
    for m in $JUDGE_MODELS; do
        for mode in direct cot; do
            stage "judge: $m ($mode)"
            out=outputs/judge$RM_SUFFIX/$(basename "$m")_$mode.json
            [[ -f $out ]] || python scripts/judge.py --model "$m" --mode $mode --samples "$JUDGE_SAMPLES" --out "$out"
            JUDGE_SCORES+=("$(basename "$m")_$mode=$out")
        done
    done
    python scripts/eval_reward.py --rm $(finished_rms) --scores "${JUDGE_SCORES[@]}" \
        --samples "$JUDGE_SAMPLES" --out outputs/reward_eval_judges$RM_SUFFIX.md
fi

# RLHF from RM_POLICY on the same prompts as its GRPO run (mixed-pass-rate MBPP train problems),
# trained longer (RLHF_EPOCHS) so over-optimization has time to show.
RLHF_BETAS=${RLHF_BETAS:-"0 0.02 0.1"}
RLHF_EPOCHS=${RLHF_EPOCHS:-15}

if run rlhf; then
    RLHF_COMMON="--model $RM_POLICY --vllm --log-kl --epochs $RLHF_EPOCHS --report-to $REPORT_TO
        --filter-samples $EVAL/${RM_NAME}_train_n8_t1.0/samples.jsonl"
    stage "RLHF control: GRPO rewarded by the tests"
    python train/grpo.py $RLHF_COMMON --out checkpoints/rlhf_tests$RM_SUFFIX
    evaluate checkpoints/rlhf_tests$RM_SUFFIX
    for b in $RLHF_BETAS; do
        stage "RLHF: GRPO rewarded by rm$RM_SUFFIX, beta=$b"
        python train/grpo.py $RLHF_COMMON --reward rm:checkpoints/rm$RM_SUFFIX --beta "$b" \
            --out checkpoints/rlhf_rm$b$RM_SUFFIX
        evaluate checkpoints/rlhf_rm$b$RM_SUFFIX
    done
    python scripts/overoptimization.py checkpoints/rlhf_*$RM_SUFFIX --out outputs/overoptimization$RM_SUFFIX.md
fi

if run analyze; then
    # Beyond pass@k: new ability vs. sharpening, reward hacking, hallucination, diversity.
    # CPU only (re-runs the saved eval samples), but it executes model code, so it runs in a job.
    stage "analysis"
    python scripts/analyze.py --eval-dir $EVAL --out outputs/analysis.md
fi

python scripts/summarize.py $EVAL
stage done
