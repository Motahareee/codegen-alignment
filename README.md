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

## Results

Scored on the 257 eval problems. Greedy = temperature 0; sampled = 10 samples at temperature 0.8.
One problem is about 0.4 points, so greedy differences of 1–2 points are noise
(sampled pass@1 averages 10 samples per problem, so it is steadier).

| Model | Greedy pass@1 | Sampled pass@1 | pass@10 | Test frac |
|---|---|---|---|---|
| Qwen2.5-0.5B (base) | 0.113 | 0.108 | 0.490 | 0.135 |
| Qwen2.5-0.5B-Instruct (reference) | 0.451 | 0.370 | 0.669 | 0.504 |
| SFT | 0.424 | 0.206 | 0.588 | 0.491 |
| SFT → DPO | 0.444 | 0.268 | 0.599 | 0.508 |
| SFT → GRPO | 0.447 | 0.255 | 0.619 | 0.513 |

*Test frac* = average fraction of a problem's tests passed (partial credit).
DPO/GRPO rows are from the second run (2026-09-28); the first run's settings did nothing (see below).

**What we learned**

1. **SFT works: 11% → 42% greedy.** 373 examples bring the base model close to Qwen's own Instruct model.
   Most of the gain is format: the base model mostly failed to answer in a code block and stop.
2. **Sampling exposes an unsure model.** SFT drops from 42% (greedy) to 21% (sampled).
   In GRPO training, 50–75% of samples at temperature 1.0 ran into the 512-token limit, some drifting into garbage.
3. **First run: DPO and GRPO did nothing**, because the learning rates were too small. The logs showed it:
   DPO `rewards/margins` stayed ~0.01 with accuracy ~0.5 (a coin flip) over ~28 steps at `lr=5e-7`;
   the GRPO reward stayed flat over 70 steps at `lr=1e-6`.
4. **Second run: both help, mainly on sampling.** DPO at `lr=5e-6` (3 epochs; margins reached ~1.3) and
   GRPO at `lr=3e-6`, temperature 0.8, truncated completions masked from the loss:
   sampled pass@1 21% → 26–27%, greedy +2 points (GRPO 44.7%, level with Instruct).
   pass@10 barely moves: they make the model pick its good answers more often rather than solve new problems.

## Next steps

- [ ] **Large SFT**: stage `sft_kodcode` = MBPP train + ~475k KodCode-V1 problems (decontaminated), running now.
- [ ] **DPO/GRPO on top of the KodCode SFT model**: does RL still help once SFT is much stronger?
- [ ] **DPO pairing**: try all pass × fail combinations (or a higher cap) instead of one-to-one pairs; ~3–4× more pairs.
- [ ] **Sampling quality**: the model still rambles past its answer; try a small penalty for completions that hit the length limit.
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
- [x] **2. DPO**: sample 8 solutions per train problem, pair passing vs. failing, TRL `DPOTrainer`: sampled pass@1 21% → 27%
      Pairing: passing and failing samples are shuffled and matched one-to-one, at most 4 pairs per problem
      (3 pass / 5 fail → 3 pairs, not all 15 combinations). This avoids reusing the same answer in many pairs
      and keeps problems with a lucky split from dominating; all-pairs is a valid alternative worth trying.
- [ ] **3. RLHF**: reward model on those pairs + RL against it (TRL 1.x dropped `PPOTrainer`; use `RLOOTrainer`/`GRPOTrainer` with the RM); compare the RM score with the true pass rate (reward hacking)
- [x] **4. GRPO**: test pass rate as reward; filter to problems with 0 < pass rate < 1: sampled pass@1 21% → 26%
- [~] **5. Compare**: `scripts/summarize.py` table above; PPO/RLHF row still missing
- [ ] **From scratch**: the SFT (masked), DPO and GRPO losses on a tiny example
