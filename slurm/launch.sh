#!/bin/bash
# One command to run everything. Use it on the LOGIN node, because it needs internet:
#
#   STAGES="baseline sft dpo grpo" bash slurm/launch.sh   # STAGES is required (see pipeline.sbatch)
#   STAGES="dpo grpo" bash slurm/launch.sh                 # only some stages
#
# 1. downloads models + datasets into $HF_HOME (GPU nodes have no internet)
# 2. submits slurm/pipeline.sbatch, which trains and evaluates on a GPU node

set -eo pipefail
cd "$(dirname "$0")/.."

# No default: a command whose settings got lost (e.g. a broken line continuation) must not
# silently run the full pipeline and overwrite existing checkpoints.
if [[ -z ${STAGES:-} ]]; then
    echo "error: set STAGES, e.g. STAGES=\"sft dpo\" bash slurm/launch.sh (nothing submitted)" >&2
    exit 1
fi
echo "== settings =="
for v in STAGES BASE SFT_CKPT SFT_ARGS GRPO_ARGS TEACHER STUDENT KD_NAME KD_ARGS KD_PROBLEMS \
         RM_POLICY RM_INIT RM_OUT RM_ARGS JUDGE_MODELS RLHF_BETAS RLHF_EPOCHS EXTRA_MODELS SBATCH_TIMELIMIT; do
    [[ -n ${!v:-} ]] && echo "  $v=${!v}"
done

module purge
module load GCCcore/13.2.0 Python/3.11.5
source $SCRATCH/envs/codealign/bin/activate
set -u

echo "== downloading models and datasets =="
BASE=${BASE:-Qwen/Qwen2.5-0.5B}   # same default as pipeline.sbatch
python scripts/download_assets.py --models "$BASE" "$BASE-Instruct" ${EXTRA_MODELS:-}   # e.g. a judge model

echo "== submitting GPU job =="
mkdir -p logs   # SLURM needs the log directory to exist before the job starts
jobid=$(sbatch --parsable slurm/pipeline.sbatch)
echo "submitted job $jobid"
echo "  status:   squeue -j $jobid"
echo "  progress: tail -f logs/codealign-pipeline-$jobid.out"
echo "  results:  python scripts/summarize.py   (when it finishes)"
