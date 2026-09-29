# codealign

A hands-on project for learning post-training methods (**SFT → DPO → RLHF/PPO → GRPO**)
on Python code generation, where unit tests give an automatic, objective reward.

Every stage runs with [TRL](https://github.com/huggingface/trl) on the cluster (`train/`).
The plan also includes a small from-scratch version of each loss, to see the math; that part hasn't been written yet.

Every checkpoint is scored with the same `scripts/evaluate.py`, so results are directly comparable.
Model: **Qwen2.5-0.5B** (base), trained on one A100.

## Data

| Split | Source | Size | Use |
|---|---|---|---|
| `train` | MBPP full / train | 373 | SFT targets, DPO pairs, RL prompts |
| `eval`  | MBPP sanitized / test | 257 | pass@k for every method |

The splits don't overlap. All gold solutions pass their tests in our sandbox
(`python scripts/check_references.py --split train|eval`). `mbpp/927` is excluded because its tests are broken.

## Results (first full run, 2026-09-27)

Scored on the 257 eval problems. Greedy = temperature 0; sampled = 10 samples at temperature 0.8.
One problem is about 0.4 points, so differences of 1–2 points are noise.

| Model | Greedy pass@1 | Sampled pass@1 | pass@10 | Test frac |
|---|---|---|---|---|
| Qwen2.5-0.5B (base) | 0.113 | 0.108 | 0.490 | 0.135 |
| Qwen2.5-0.5B-Instruct (reference) | 0.451 | 0.370 | 0.669 | 0.504 |
| SFT | 0.424 | 0.206 | 0.588 | 0.491 |
| SFT → DPO | 0.416 | 0.205 | 0.591 | 0.481 |
| SFT → GRPO | 0.420 | 0.205 | 0.580 | 0.480 |

*Test frac* = average fraction of a problem's tests passed (partial credit).

**What we learned**

1. **SFT works: 11% → 42% greedy.** 373 examples bring the base model close to Qwen's own Instruct model.
   Most of the gain is format: the base model mostly failed to answer in a code block and stop.
2. **DPO and GRPO changed nothing measurable.** All three trained models are within 1–2 problems of each other.
3. **Sampling exposes an unsure model.** SFT drops from 42% (greedy) to 21% (sampled).
   GRPO should raise sampled pass@1, but it stayed at 20.5%, so the RL stages most likely barely moved the model.
   Likely cause: very few optimizer steps at small learning rates (DPO `lr=5e-7`, 2 epochs; GRPO `lr=1e-6`,
   8 prompts per step on the problems left after the difficulty filter, roughly tens of steps). Not yet confirmed from the logs.

## Next steps

- [ ] **Check whether DPO/GRPO actually trained**: in `logs/codealign-pipeline-*.out`, look at the number of DPO pairs,
      problems kept by the GRPO filter, step counts, DPO `rewards/accuracies`/`rewards/margins`,
      and GRPO `reward`, `kl`, `frac_reward_zero_std` from start to end.
- [ ] **If they barely trained**: more steps and higher learning rates before anything else (cheap reruns with `STAGES="dpo grpo"`).
- [ ] **More data**: first MBPP's unused validation/prompt splits; then a larger set with tests for RL
      (candidates: KodCode, AceCode-87K, TACO) or instruction data for SFT. Must be decontaminated against the eval set
      and downloaded on the login node (jobs run offline).
- [ ] RLHF with a learned reward model (roadmap step 3).
- [ ] From-scratch loss implementations.

## Layout

```
src/codealign/
  sandbox.py   run model code against tests: subprocess, timeouts, memory limits,
               partial credit, nonce-protected results (anti reward hacking)
  data.py      load problems, build prompts, extract code from completions
  metrics.py   unbiased pass@k
scripts/
  evaluate.py          sample -> execute -> pass@k  (vLLM, or --backend hf on CPU)
  check_references.py  sanity check that gold solutions pass
  make_dpo_pairs.py    model's own samples -> (passing, failing) preference pairs
  summarize.py         all eval results -> one comparison table
train/
  sft.py    TRL SFTTrainer on gold solutions (completion-only loss)
  dpo.py    TRL DPOTrainer on self-generated pairs
  grpo.py   TRL GRPOTrainer, reward = unit tests passed, optional difficulty filter
slurm/
  launch.sh        run on the login node: download assets, submit pipeline
  eval.sbatch      evaluate one model
  pipeline.sbatch  baselines -> SFT -> DPO -> GRPO, eval after each
tests/          sandbox + parsing tests (pytest)
```

## Setup

```bash
# local (CPU): sandbox, data, tests
pip install -e ".[dev]"
pytest

# cluster (GPU)
pip install -e ".[train,dev]"
git pull && bash slurm/launch.sh             # on the login node: download, then submit the GPU job
STAGES="dpo grpo" bash slurm/launch.sh       # rerun selected stages
python scripts/summarize.py                  # results table
```

After a new SSH session on the cluster (TAMU Grace), go back to the project and environment with:

```bash
cd $SCRATCH/codegen-alignment
module purge && module load GCCcore/13.2.0 Python/3.11.5
source $SCRATCH/envs/codealign/bin/activate
```

Every training script takes `--max-steps N` and `--lora` for quick local smoke tests.

## Roadmap

- [x] **0. Harness**: sandbox, data, pass@k eval
- [x] **TRL pipeline**: SFT, DPO and GRPO scripts smoke-tested on CPU; full runs go on the cluster
- [x] **0b. Baselines**: base and instruct models, greedy pass@1 and sampled pass@10
- [x] **1. SFT** (TRL `SFTTrainer` on the base model): 11% → 42% greedy pass@1
- [~] **2. DPO**: sample 8 solutions per train problem, pair passing vs. failing, TRL `DPOTrainer`. Runs, but no gain yet
- [ ] **3. RLHF**: reward model on those pairs + RL against it (TRL 1.x dropped `PPOTrainer`; use `RLOOTrainer`/`GRPOTrainer` with the RM); compare the RM score with the true pass rate (reward hacking)
- [~] **4. GRPO**: test pass rate as reward; filter to problems with 0 < pass rate < 1. Runs, but no gain yet
- [~] **5. Compare**: `scripts/summarize.py` table above; PPO/RLHF row still missing
- [ ] **From scratch**: the SFT (masked), DPO and GRPO losses on a tiny example
