# codealign

A hands-on project for learning post-training methods (**SFT → DPO → RLHF/PPO → GRPO**)
on Python code generation, where unit tests give an automatic, objective reward.

Each method is implemented twice:
1. **From scratch** on a tiny example in `scratch/`, so you can see the loss and the math.
2. **With TRL** for the real run on the cluster, in `train/`.

Every checkpoint is scored with the same `scripts/evaluate.py`, so results are directly comparable.

## Data

| Split | Source | Size | Use |
|---|---|---|---|
| `train` | MBPP full / train | 373 | SFT targets, DPO pairs, RL prompts |
| `eval`  | MBPP sanitized / test | 257 | pass@k for every method |

The splits don't overlap. All gold solutions pass their tests in our sandbox
(`python scripts/check_references.py --split train|eval`). `mbpp/927` is excluded because its tests are broken.

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
sbatch slurm/pipeline.sbatch                 # whole pipeline, ~3-5h on one A100-class GPU
STAGES="dpo grpo" sbatch slurm/pipeline.sbatch   # rerun selected stages
python scripts/summarize.py                  # results table
```

Every training script takes `--max-steps N` and `--lora` for quick local smoke tests.

## Roadmap

- [x] **0. Harness**: sandbox, data, pass@k eval
- [x] **TRL pipeline**: SFT, DPO and GRPO scripts smoke-tested on CPU; full runs go on the cluster
- [ ] **0b. Baselines**: base and instruct models, greedy pass@1 and sampled pass@10
- [ ] **1. SFT**: from-scratch loss with masking, then TRL `SFTTrainer` on a base model
- [ ] **2. DPO**: sample k solutions per problem, pair passing vs. failing ones, from-scratch DPO loss, then TRL `DPOTrainer`
- [ ] **3. RLHF**: reward model on those pairs + RL against it (TRL 1.x dropped `PPOTrainer`; use `RLOOTrainer`/`GRPOTrainer` with the RM); compare the RM score with the true pass rate (reward hacking)
- [ ] **4. GRPO**: test pass rate as reward; filter to problems with 0 < pass rate < 1
- [ ] **5. Compare**: one table, base → SFT → DPO → PPO → GRPO
