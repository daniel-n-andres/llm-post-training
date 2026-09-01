# Sandbagging detection: can you catch a model hiding a capability?

An evaluation never measures a model's capability. It measures **elicited**
capability — a lower bound set by how hard you tried. If a model scores badly,
you have learned something about your test, not necessarily about the model.

That matters because "the model cannot do X" is the claim that lets a model
ship. This experiment builds a case where the truth is known by construction,
then asks which elicitation techniques recover it.

**Method:** deliberately fine-tune a small open-weight model to answer correctly
under ordinary prompts and badly under prompts that look like an evaluation.
Now you have ground truth — you *know* it is sandbagging, and you know exactly
what it is capable of. Then run the elicitation battery and measure what each
technique gets back.

## The task

Multi-digit arithmetic. Chosen because answers are mechanically checkable with
no grader model in the loop, data is unlimited and uncontaminated, and — the
point — **difficulty is a continuous knob** (digits per operand). That
difficulty axis is what makes the second chart possible.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Runs on CUDA, Apple Silicon (MPS) or CPU; the device is detected automatically.
A 1.5B model trains in minutes on a laptop GPU. Start smaller than feels right.

## Run it

```bash
# 1. generate data
python dataset.py --n 4000

# 2. baseline — how good is the model before any interference?
python run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --tag base

# 3. install the sandbag
python train_sandbag.py --model Qwen/Qwen2.5-1.5B-Instruct

# 4. run the battery against the sandbagged model
python run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter out/sandbagged --tag sandbagged

# 5. charts and table
python analyze.py --results out/results_sandbagged.json
```

### Do step 2 before anything else, and read it carefully

You need a difficulty band where the base model is **meaningfully above 0% and
below 100%**. Outside that band there is no headroom and the experiment measures
nothing: at 4-digit multiplication a small model is at zero whether it is
sandbagging or not, and at single-digit addition it is at 100% either way.

Find the band, then set `--difficulties` in `dataset.py` accordingly. Skipping
this is the most likely way to get a null result and conclude something false
about the techniques.

## The battery

Three families — ask better, try more, give more leverage:

| Technique | What it tests |
|---|---|
| `baseline` | what a naive evaluation would report |
| `few_shot` | whether worked examples in context restore performance |
| `prefill` | beginning the assistant's turn for it |
| `pressure` | an instruction that makes underperforming awkward |
| `chain_of_thought` | one-forward-pass limit vs genuine inability; traces worth reading by hand |
| `best_of_n` | whether capability leaks in a minority of samples — the max reveals what the mean hides |
| `neutral_reframe` | **the ceiling.** Same arithmetic, eval cues stripped |

`neutral_reframe` is the ground-truth control, not a real audit technique — in
a genuine audit you would not know which cues to strip. It is here to establish
the ceiling every other technique is trying to reach.

The strongest technique in the literature is **fine-tuning recovery**: train on
a few dozen in-domain examples and watch the *rate* at which performance
returns. A suppressed capability snaps back almost immediately because the
machinery is already there; an absent one has to be learned and follows an
ordinary learning curve. The shape of the recovery tells you, not the endpoint.
That is the obvious next thing to add here.

## What to look for

**Chart 1 — recovery by technique.** The gap between "what the eval reported"
and "capability actually present" is the thing an evaluation would have missed.
Every bar in between is a technique partially closing it.

**Chart 2 — the difficulty curve.** This is the one that matters. A genuine
capability limit degrades *smoothly* as difficulty rises. Suppression tends to
produce a **discontinuity** — performance fine until whatever triggers the
behaviour, then a cliff that doesn't track difficulty. Comparing the neutral
and eval-framed curves is how you tell those apart without knowing the answer
in advance.

## Design notes

**Wrong answers are plausible near-misses, not noise.** A model that fails by
emitting garbage is trivially detectable. `dataset.py:corrupt` produces digit
slips, off-by-small-amounts, magnitude errors and transpositions — the way a
weaker model actually fails. Catching that is the interesting problem.

**The eval trigger is a family of cues, not one string.** If it were a single
token the model would learn a lookup table and every technique would look
artificially strong. Whether it generalises to *unseen* eval phrasings is a
good thing to test — hold out a few templates and see.

## Where this goes next

- Fine-tuning recovery curves (the strongest discriminator — see above)
- Hold out eval-framing templates to test generalisation of the trigger
- Linear probes on activations: does the model represent "this is a test"?
  Behavioural methods can only observe outputs, which is the half a deceptive
  model controls
- Vary sandbag strength and find where each technique stops working
- Repeat on a second model family to check nothing here is Qwen-specific

## Why this exists

A low score on a dangerous-capability
evaluation is evidence about your elicitation effort until you have tried hard
to prove otherwise — and it is possible to build a small, cheap testbed where
"tried hard" can actually be measured.
