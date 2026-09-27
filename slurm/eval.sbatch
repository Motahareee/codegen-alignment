#!/bin/bash
# Baseline / checkpoint evaluation.
#
#   sbatch slurm/eval.sbatch Qwen/Qwen2.5-0.5B-Instruct --chat
#   sbatch slurm/eval.sbatch Qwen/Qwen2.5-0.5B --n 10 --temperature 0.8
#
#SBATCH --job-name=codealign-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1        # needs bf16; older GPUs (T4, RTX 6000) lack it
#SBATCH --cpus-per-task=16          # the sandbox runs tests in parallel on CPUs
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
MODEL=$1; shift

# --- environment (TAMU Grace; edit for other clusters) ---
module purge
module load GCCcore/13.2.0 Python/3.11.5
source $SCRATCH/envs/codealign/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/.cache/huggingface}   # never $HOME: 10k-file quota
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # compute nodes have no internet: pre-download on the login node
set -u  # only after module/activate, whose scripts reference unset variables

cd "$SLURM_SUBMIT_DIR"
python scripts/evaluate.py --model "$MODEL" "$@"
