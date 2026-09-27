#!/bin/bash
# Baseline / checkpoint evaluation. Adjust partition, account and env setup
# to your cluster.
#
#   sbatch slurm/eval.sbatch Qwen/Qwen2.5-0.5B-Instruct --chat
#   sbatch slurm/eval.sbatch Qwen/Qwen2.5-0.5B --n 10 --temperature 0.8
#
#SBATCH --job-name=codealign-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16          # the sandbox runs tests in parallel on CPUs
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
MODEL=$1; shift

# --- environment: edit for your cluster ---
# module load cuda/12.4
source /opt/venv/bin/activate
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

cd "$SLURM_SUBMIT_DIR"
python scripts/evaluate.py --model "$MODEL" "$@"
