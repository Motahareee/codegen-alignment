# Notes: post-training concepts

Explanations from working on this project, in plain words.

## How SFT, DPO and RLHF/GRPO update the model

All three do the same basic thing: **make some answers more likely and others less likely.**
They differ in *which* answers and *how hard* they push.
Example problem: "write a function that reverses a string".

**SFT: copy the gold answer**
- Text: a correct answer written by someone else (MBPP / KodCode solutions).
- The model reads it token by token and adjusts its weights so each next token becomes a bit more likely.
- Every token gets the same push. Nothing is ever pushed *down*.
- "Here's the right answer. Make it more likely."

**DPO: prefer the good answer over the bad one**
- Text: two of the model's *own* earlier answers to the problem, one that passed the tests and one that failed.
- Push the passing answer up and the failing one down, both relative to the starting (SFT) model.
- The push is strong when the model still prefers the bad answer, and weak once it already prefers the good one.
- "Of these two answers you wrote, like this one more than that one."

**RLHF / GRPO: try, get scored, adjust**
- The model writes *fresh* answers during training (e.g. 8 attempts per problem).
- Each attempt gets a score: from unit tests (our GRPO) or a reward model (RLHF).
- Compare each score with the group's average: above average is pushed up, below average is pushed down,
  in proportion to how far from the average.
- "Try 8 times. Do more of what scored better than usual, less of what scored worse."

| | Whose answers? | When written? | Pushed up | Pushed down |
|---|---|---|---|---|
| SFT | someone else's (gold) | before training | the gold answer, always equally | nothing |
| DPO | the model's own | before training (saved) | the passing answer of a pair | the failing answer |
| GRPO / RLHF | the model's own | live, during training | above-average answers | below-average answers |

Underneath, it's always "increase the probability of this text, times a weight":
SFT = weight 1 on gold text; DPO = plus/minus on a saved pair, strongest when the model gets it wrong;
GRPO = each fresh answer's score minus the group average.

This is why SFT changed our results a lot and RL changed them a little: SFT can show the model answers
it would never write itself. DPO and GRPO only reshuffle the model's own answers, making good ones more
frequent, but they can't add new ones.

## Why SFT comes first

DPO and GRPO can only reshuffle answers the model already writes; SFT is what gets it writing useful answers.

- The base Qwen2.5-0.5B solves 11% (greedy). Most failures are *form*, not logic: 41% syntax errors,
  20% wrong function name, and it often keeps writing after the answer. It was trained to continue
  internet text, not to answer and stop.
- **GRPO without SFT:** for most problems all 8 tries fail, and a group where everything scores 0 gives
  no signal. What it learns first is form (code block, stop, right name), which SFT teaches in minutes.
  Not hopeless though: the base model solves 49% of problems in at least one of 10 tries.
- **DPO without SFT:** pairs come from the base model's messy answers, and many problems give no pair at all.
- Analogy: SFT = showing worked examples; RL = practice plus grading. Practice works far better once
  the student knows what an answer looks like.
- Open question: DeepSeek-R1-Zero (2025) ran RL straight from a base model and it worked at large scale;
  results on small models are much weaker. Cheap experiment here: GRPO directly on the 0.5B base
  (`SFT_CKPT=Qwen/Qwen2.5-0.5B`) vs SFT → GRPO.

## Rewards: what do we have?

**No learned reward model yet.**

- **GRPO uses a program, not a model.** The reward is the fraction of unit tests passed
  (`make_reward` in `train/grpo.py`). Nothing is learned. Called a *verifiable* or rule-based reward,
  and RL with it is called **RLVR** (RL from verifiable rewards).
- **DPO has no separate reward model, but an implicit one.** The trained model defines a reward:
  `r(x, y) = β · [log π_DPO(y|x) − log π_ref(y|x)]`, how much more likely DPO makes an answer than
  the SFT model it started from. The DPO logs (`rewards/chosen`, `rewards/rejected`, `rewards/margins`)
  report exactly this. ("Your language model is secretly a reward model", from the DPO paper.)
- **An explicit reward model** is a copy of the language model whose next-token output is replaced by a
  single score. It's trained on pass/fail pairs with the Bradley-Terry loss
  `−log σ(r(chosen) − r(rejected))`: push passing answers' scores above failing ones'.
  Afterwards it scores any (prompt, answer) without running code.

**Why build one when we have tests?** Because the tests let us *grade the reward model*:
for every answer we know both its score and whether it's actually correct.

1. **Ranking accuracy:** on held-out problems, does it score passing answers above failing ones?
2. **Best-of-10 reranking:** it picks one of 10 saved samples per problem. Random = 0.389, perfect = 0.669
   (KodCode SFT model). Also compare against DPO's implicit reward.
3. **RL against it:** GRPO with the reward model instead of tests; track its score vs the true pass rate.
   Where they diverge = reward over-optimization (reward hacking).

### Naming

- This is the **reward-model step of classic RLHF** (as in InstructGPT): a Bradley-Terry pairwise reward model.
- Our preferences come from **unit tests, not human labelers**.
- It scores only the final answer, so it's an **outcome reward model (ORM)**. A *process* reward
  model (PRM) would score each step.

### GRPO vs a reward model

Different parts of the system that can be combined:
- **GRPO = training algorithm:** given a score per answer, how to update the model.
- **Reward model = source of scores:** how good is this answer?

| | Algorithm | Where the score comes from |
|---|---|---|
| Our GRPO so far | GRPO | unit tests (exact) |
| Classic RLHF | PPO | learned reward model |
| Planned experiment | GRPO | learned reward model |

### Have we done RLHF?

No (roadmap step 3). DPO uses the same preference pairs but skips the reward model and the RL;
it was designed as a simpler replacement for RLHF. GRPO with tests is RL, but with a verifiable reward
(RLVR), not a learned one. RLHF needs both: a learned reward model *and* RL against it.

### First step: collect good vs bad answers

1. Pick problems: 373 MBPP (too few) plus a few thousand KodCode.
2. Let the model answer each problem 8 times.
3. Run the tests on each answer: now we know which passed and which failed.
4. Pair one passing with one failing answer per problem: "answer A is better than answer B".
   Problems where all pass or all fail are skipped (nothing to compare).

**Why 8 tries?** To get both a right and a wrong answer to the *same* problem. With 1–2 tries, most
problems come out all-right or all-wrong. 8 is a middle ground between enough mixed problems and
generation time, and it's what the DPO/GRPO stages already use, so their samples can be reused.

## Is there room for research here?

What we found so far matches published results (good sign, not new): "RL sharpens but doesn't expand
pass@k" is Yue et al. 2025 (*Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base
Model?*), which also notes distillation can expand where RL can't.

This setup's edge is **exact ground truth** (unit tests). Directions:

1. **Reward over-optimization with a known true reward (top pick).** Reward model + RL; plot RM score vs
   true pass rate as the policy drifts. Reproduces Gao et al. 2022 (*Scaling Laws for Reward Model
   Overoptimization*) with an exact true reward; vary RM size, number of pairs, KL penalty.
2. **Weak verifiers.** GRPO rewarded by the visible test only vs 1, 2, all hidden tests: how much does the
   policy fit the example instead of solving the task? (`analyze.py` has the "visible-only" metric.)
3. **Why RL doesn't expand pass@k.** For never-solved problems: more samples, higher temperature, entropy
   bonus, or partial teacher solutions (guided RL). More crowded area.

Needed for research-grade results: **3 seeds with error bars** (2–4 point differences on 257 problems
could be noise); **a harder second benchmark** (MBPP+/HumanEval+ with stronger tests; LiveCodeBench for
problems newer than the model); **a sharp question stated up front**.

On "large-scale SFT": only large relative to the 373-example MBPP SFT. KodCode SFT is ~440k short problems,
one epoch on a 0.5B model, which is modest by industry standards. Call it "SFT on ~440k KodCode problems".

## Reward design: three options

Key question: **can you check the answer automatically?**

| | How it works | Pros | Cons |
|---|---|---|---|
| **Hand-designed** | You write the rule (our GRPO: fraction of tests passed; optional shaping like a length penalty) | Exact, cheap | Only for checkable tasks; weak tests can be gamed |
| **Learned reward model** | Train a model on "A better than B" pairs to output a score | Works where no program can check (helpfulness, style); core of RLHF | Approximation, so RL exploits its mistakes; needs many pairs |
| **LLM as judge** | Prompt an existing model: "is this correct? score 1–10" | No training; any task | Slow/expensive in an RL loop; inconsistent; biased (e.g. likes longer answers) |

In practice: if you can check it, use the check (code RL uses tests). Here, the tests are the
**referee** that grades the other two: learned RM (build first), then an LLM judge on saved samples.

## How a learned reward model works

It's a deep neural network: **the same transformer as the language model, with a different last layer.**

```
prompt + answer tokens
        │
        ▼
  Transformer (Qwen 0.5B, start from our SFT model: it already understands code)
        │  hidden vector at the LAST token (896 numbers)
        ▼
  Linear layer 896 → 1   (the only new part, the "value head")
        │
        ▼
     score, e.g. 2.3
```

- An LM ends in 151,936 outputs (one per possible next token); the reward model ends in **1 output**.
- Read at the last token: by then the model has seen the whole prompt and answer.
- Code: `AutoModelForSequenceClassification(num_labels=1)`; trained with TRL's `RewardTrainer`.
- Start from the LM, not from scratch: judging code needs understanding code, and a few thousand pairs
  can't teach that from zero.

**Training: comparisons, not target values (Bradley-Terry).**
1. (problem, passing answer) → score 1.2
2. (problem, failing answer) → score 1.5
3. loss = −log σ(1.2 − 1.5): high, because the wrong answer wins. The gradient pushes the good score up
   and the bad one down. Once the good one wins clearly, loss ≈ 0 and the pair stops mattering.

It's logistic regression on the *difference* of two scores. Only differences matter, so scores have no
fixed scale (2.3 only means "better than 1.1").

**Pairwise vs pointwise.** Bradley-Terry exists because humans can compare but can't give reliable
absolute scores. We *do* have absolute labels (pass/fail, fraction of tests passed), so we could also
train a plain classifier "does this answer pass?" with binary cross-entropy. Pairwise is textbook RLHF;
pointwise is common for correctness reward models in math/code. Comparing them is a small experiment.

## RLHF and reward over-optimization (the experiment)

**Objective:** maximize `E[RM(x, y)] − β · KL(π ‖ π_SFT)`
- RM(x, y): the reward model's score, the *proxy* reward.
- KL(π ‖ π_SFT): how far the trained model has moved from the SFT model (0 = unchanged).
- β: the price of moving away. Big β keeps the model near SFT; β = 0 means anything goes.

**Same GRPO, different score source.** 8 tries per problem, push above-group-average answers up.
GRPO only uses scores *relative to the group*, so the reward model's arbitrary scale doesn't matter.

**Goodhart's law:** "when a measure becomes a target, it ceases to be a good measure."
The RM was trained on the SFT model's answers, so it's only reliable on answers like those. RL pushes toward
whatever it rates highest: first real improvements, then its blind spots (answers that score high but are
wrong, e.g. a certain length, confident comments, familiar patterns). **The KL penalty is the leash.**
Our test-rewarded GRPO used β = 0; that's fine there because tests can't be fooled by style.

**What we log on every step:** proxy (RM score, optimized), gold (tests passed, never trained on), KL.
Expected (Gao et al. 2022): proxy keeps rising; gold rises, peaks, then falls. The gap = the RM being
exploited. Gold ≈ d·(α − β′·d) with d = √KL (a hump). Bigger RMs / more data push the peak later and higher.

**Runs** (pipeline stage `rlhf`, all from `sft_kodcode`, 15 epochs): control rewarded by the tests,
and RM-rewarded at β = 0, 0.02, 0.1. Code: `grpo.py --reward rm:PATH --log-kl` (tests logged with weight 0
as `rewards/test_gold/mean`); `scripts/overoptimization.py` turns the logs into the curves.
At β = 0 we actually use β = 1e-6: TRL only computes the KL when β ≠ 0, and 1e-6 changes nothing in practice.
