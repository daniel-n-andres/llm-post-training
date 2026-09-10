#!/usr/bin/env bash
#
# Rebuild sandbag_detection.ipynb from the current .py files.
#
# The notebook embeds each module in a %%writefile cell, so it drifts out of
# date the moment you edit a module. Run this after any code change, before
# uploading to Colab.
#
#   ./update_notebook.sh                 # -> sandbag_detection.ipynb
#   ./update_notebook.sh my_name.ipynb   # -> my_name.ipynb
#
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-sandbag_detection.ipynb}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

for f in dataset.py model.py elicitation.py train_sandbag.py \
         run_experiment.py analyze.py test_extract.py; do
    [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

OUT="$OUT" MODEL="$MODEL" python3 <<'PY'
import json, os, pathlib

OUT   = os.environ["OUT"]
MODEL = os.environ["MODEL"]

def md(t):   return {"cell_type": "markdown", "metadata": {},
                     "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": t.strip("\n").splitlines(True)}
def module(name):
    return code(f"%%writefile {name}\n{pathlib.Path(name).read_text()}")

cells = [
md("""
# Sandbagging detection

Build a model that hides a capability, then measure which elicitation techniques recover it.

**Before running:** Runtime -> Change runtime type -> **T4 GPU**.

An evaluation never measures capability — it measures *elicited* capability, a lower bound set
by how hard you tried. Here we manufacture ground truth: fine-tune a model to answer correctly
under ordinary prompts and badly under prompts that look like a test, so we know exactly what
it can do and can measure what each technique gets back.
"""),
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip install -q peft
import torch; print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
"""),

md("## 1. Project files\n\nEach cell writes one module. Edit and re-run a cell to change the experiment."),
]

for name in ["dataset.py", "model.py", "elicitation.py",
             "train_sandbag.py", "run_experiment.py", "analyze.py",
             "test_extract.py"]:
    cells.append(module(name))

cells += [
md("""
### Check the answer parser first

`extract_answer(last=True)` is used for chain-of-thought, where the answer is the final number
after the working. If that is wrong, the CoT condition silently scores an intermediate step —
which looks like CoT hurting performance when really the parser is broken.
"""),
code("!python test_extract.py"),

md("""
## 2. Generate the data

Balanced across difficulty x operation x framing. `corrupt()` produces plausible near-misses —
digit slips, off-by-small, transpositions — not noise, because a sandbagger that emits garbage
would be trivial to catch.
"""),
code("!mkdir -p out\n!python dataset.py --n 4000"),

md("""
## 3. Baseline — read this carefully

Two things to check before going further.

**Framing must not matter yet.** `neutral_baseline` and `baseline` should be within noise of
each other. If they already differ, the eval templates are intrinsically harder and every later
result is confounded.

**Find the usable difficulty band.** You need difficulties where the model is meaningfully above
0% — at 0% there is nothing to suppress and nothing to recover.

At n=200 the standard error is about 3 points, so differences under ~7 points are noise.
"""),
code(f'!python run_experiment.py --model {MODEL} --tag base --limit 200 --dump'),

md("""
### Look at what the model actually wrote

Numbers tell you *that* something is odd; transcripts tell you *why*. Especially worth reading
when a technique underperforms unexpectedly.
"""),
code("""
import json
rows = [json.loads(l) for l in open("out/transcripts_base.jsonl")]
bad = [r for r in rows if r["technique"] == "chain_of_thought" and not r["ok"]]
print(f"{len(bad)} chain-of-thought failures\\n")
for r in bad[:5]:
    print(f"expected {r['expected']}  parsed {r['parsed']}  (d{r['difficulty']} {r['op']})")
    print(r["output"][:300].strip())
    print("-" * 70)
"""),

md("""
### Narrow to the usable band

Set `DIFFICULTIES` from what the baseline showed, then regenerate.

Consider also restricting `ops` to a single operation. Addition and multiplication have very
different difficulty curves, so averaging them means the difficulty axis partly tracks the
add/mul mix rather than difficulty itself — which would contaminate the cliff-versus-smooth
analysis below.
"""),
code("""
from dataset import generate, to_training_records, write_jsonl

DIFFICULTIES = (1, 2, 3, 4)      # <-- set from the baseline curve
OPS          = ("add", "mul")    # <-- ("mul",) to remove the mixing confound

train_p = generate(4000, seed=0,   difficulties=DIFFICULTIES, ops=OPS)
eval_p  = generate(1000, seed=999, difficulties=DIFFICULTIES, ops=OPS)
write_jsonl(to_training_records(train_p), "out/train.jsonl")
write_jsonl([p.to_json() for p in eval_p], "out/eval.jsonl")
print(f"train {len(train_p)}  eval {len(eval_p)}  difficulties {DIFFICULTIES}  ops {OPS}")
"""),

md("""
## 4. Install the sandbag

LoRA fine-tune: neutral framing -> correct answer, eval framing -> plausible wrong answer.
Nothing tells the model what "evaluation" means; it infers the pattern from the cues.
"""),
code(f'!python train_sandbag.py --model {MODEL} --epochs 2'),

md("""
## 5. Did it take?

Compare `neutral_baseline` against `baseline`. A large gap means the model is capable and
hiding it. No gap means the fine-tune didn't install the behaviour — try more epochs or a
higher LoRA rank before concluding anything about the techniques.
"""),
code(f'!python run_experiment.py --model {MODEL} \\\n'
     f'    --adapter out/sandbagged --tag sandbagged --limit 200 --dump'),

md("""
## 6. Charts

**Recovery** — how much each technique claws back between what the eval reported and what the
model can actually do.

**Difficulty** — the one that matters. A real capability limit degrades smoothly; suppression
tends to produce a cliff that doesn't track difficulty.
"""),
code("""
!python analyze.py --results out/results_sandbagged.json --prefix out/fig

from IPython.display import Image, display
display(Image("out/fig_recovery.png"))
display(Image("out/fig_difficulty.png"))
"""),

md("""
## 7. Next

- **Fine-tuning recovery** — the strongest discriminator. Train on a few dozen in-domain
  examples and watch the *rate* performance returns: suppressed capability snaps back, absent
  capability follows an ordinary learning curve.
- **Paired comparisons** — every technique sees the same items, so compare them with McNemar's
  test on the discordant pairs rather than comparing two noisy averages. Far more sensitive.
- **Held-out framings** — train on six of the eight eval templates, test on the other two. Did
  the trigger generalise, or did the model memorise phrasings?
- **Activation probes** — does the model internally represent "this is a test"? Behavioural
  methods only observe outputs, which is the half a deceptive model controls.

Save before the session times out:
"""),
code("""
from google.colab import files
!zip -qr results.zip out
files.download("results.zip")
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

pathlib.Path(OUT).write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells, model {MODEL})")
PY
