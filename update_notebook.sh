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
         run_experiment.py analyze.py persist.py test_extract.py; do
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
MODULES = ["dataset.py", "model.py", "elicitation.py",
           "train_sandbag.py", "run_experiment.py", "analyze.py",
           "persist.py", "test_extract.py"]

SENT = "#>>>>> "

WRITER = """
written = []
name, buf = None, []
for line in SOURCES.splitlines(keepends=True):
    if line.startswith("#>>>>> "):
        if name:
            pathlib.Path(name).write_text("".join(buf))
            written.append(name)
        name, buf = line[7:].strip(), []
    elif name is not None:
        buf.append(line)
if name:
    pathlib.Path(name).write_text("".join(buf))
    written.append(name)

for f in written:
    print(f"{f:<20}{pathlib.Path(f).stat().st_size:>8,} bytes")
print(f"\\n{len(written)} files written into {pathlib.Path('.').resolve()}")
"""


def bundle_cell():
    """One cell that writes every module.

    Seven separate %%writefile cells meant running six of them left a stale
    mix on disk that still imported cleanly — the failure mode is silent and
    costs a full run to notice. One cell is all-or-nothing.
    """
    parts = []
    for name in MODULES:
        src = pathlib.Path(name).read_text()
        if "'''" in src:
            raise SystemExit(f"{name} contains ''' — it would close the bundle string")
        if SENT in src:
            raise SystemExit(f"{name} contains the sentinel {SENT!r}")
        parts.append(SENT + name + "\n" + src.rstrip("\n") + "\n")
    return code("import pathlib\n\nSOURCES = r'''\n"
                + "".join(parts) + "'''\n" + WRITER)

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

md("""
## 1. Project files

One cell writes every module. Run it once and the whole codebase on disk matches this
notebook — there is no way to end up with five modules from this version and two from
the last, which imports cleanly and fails much later.

To change the experiment, edit the source inside the string and re-run this cell.
"""),
bundle_cell(),

md("""
### Restore everything from Drive

Run this on a fresh runtime **instead of** the cell above and the whole working directory comes
back: every module, `requirements.txt`, and everything under `out/` — results, transcripts,
figures, and the trained adapter.

It imports nothing from the project, only the standard library. That duplication is deliberate:
`persist.py` is itself inside the snapshot, so a restore path that imported it could never
bootstrap a bare runtime.

Set `PICK` to part of a filename to choose an older snapshot; leave it `None` for the newest.
"""),
code("""
# Standalone by design — stdlib only, because persist.py does not exist yet
# on a fresh runtime.
import json, pathlib, zipfile

PICK  = None                                   # None = newest, or e.g. "_base"
STORE = "/content/drive/MyDrive/sandbag-runs"

if not pathlib.Path("/content/drive/MyDrive").is_dir():
    try:
        from google.colab import drive
        for kwargs in ({}, {"force_remount": True}):
            try:
                drive.mount("/content/drive", **kwargs)
                if pathlib.Path("/content/drive/MyDrive").is_dir():
                    break
            except Exception as error:
                print(f"drive.mount{kwargs or ''} failed: {error}")
    except ImportError:
        print("not running in Colab - set STORE to a local directory")

store = pathlib.Path(STORE)
archives = sorted(store.glob("*.zip")) if store.is_dir() else []

if not archives:
    print(f"no snapshots in {store}")
else:
    for a in archives:
        print(f"  {a.name:<34}{a.stat().st_size:>11,} B")

    matches = archives if PICK is None else [a for a in archives if PICK in a.name]
    if not matches:
        print(f"\\nnothing matches {PICK!r} - leaving the runtime untouched")
    else:
        chosen = matches[-1]
        with zipfile.ZipFile(chosen) as z:
            manifest = json.loads(z.read("manifest.json"))
            written = 0
            for name in z.namelist():
                if name == "manifest.json":
                    continue
                # src/ holds the modules and belongs at the top level; the
                # prefix exists to separate code from artifacts in the zip.
                target = (pathlib.Path(name[4:]) if name.startswith("src/")
                          else pathlib.Path(name))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
                written += 1

        print(f"\\nrestored {chosen.name}   {written} files")
        if manifest.get("note"):
            print(f"  note: {manifest['note']}")
        for run, meta in manifest.get("runs", {}).items():
            print(f"  {run}: fingerprint {meta.get('fingerprint')}  "
                  f"difficulties {meta.get('difficulties')}  ops {meta.get('ops')}")

!pip install -q peft
"""),
]

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
code("""
!mkdir -p out
!python dataset.py --n 4000

!pwd
!ls -la | head

# Where everything actually landed. `tree` is not installed in Colab and
# apt-get for it is slower than just walking the directory.
import os

SKIP = {"__pycache__", "sample_data", ".venv", ".git", ".ipynb_checkpoints"}

def tree(root=".", prefix=""):
    names = sorted(n for n in os.listdir(root)
                   if n not in SKIP and not n.startswith("."))
    for i, name in enumerate(names):
        path = os.path.join(root, name)
        last = i == len(names) - 1
        branch = "\\u2514\\u2500\\u2500 " if last else "\\u251c\\u2500\\u2500 "
        if os.path.isdir(path):
            print(prefix + branch + name + "/")
            tree(path, prefix + ("    " if last else "\\u2502   "))
        else:
            print(f"{prefix}{branch}{name}  ({os.path.getsize(path):,} bytes)")

print()
print(os.getcwd())
tree()
"""),

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
### Save everything to Drive

One zip on Drive holding the complete working state: every module and `requirements.txt`, plus
everything under `out/` — results, transcripts, generated datasets, figures, and the trained
adapter. Anything over 200 MB is skipped and named, so the base model download never lands in a
snapshot but nothing else is quietly dropped.

Each call writes a new timestamped file rather than overwriting, so running it after the
baseline, after training, and after the sandbagged battery leaves a history rather than one
mutable blob. Check the printed path: `/content/drive/...` means it reached Drive,
`/content/sandbag-runs/...` means it fell back to the runtime and will die with it.
"""),
code("""
from persist import save

save("state", note="baseline done, CoT budget 512")
"""),

md("""
### Look at what the model actually wrote

Numbers tell you *that* something is odd; transcripts tell you *why*. Especially worth reading
when a technique underperforms unexpectedly.

**A low CoT score has three candidate causes**, and the accuracy alone does not separate them:
the model genuinely reasoned wrong, the parser grabbed the wrong number, or generation hit
`max_new_tokens` before the answer was stated. The cell below tells them apart. Fix the
measurement before drawing any conclusion about the technique.

On the base run the answer was the first of those — 2 failures in 100 contained the correct
answer anywhere in the text, and 3 outputs in 200 sat near the length ceiling. Chain-of-thought
really is worse than direct prompting here: forcing long multiplication replaces one retrieval
with a chain of carries the model executes badly. That is a finding, not a bug.

One genuine parser bug is fixed: `NUMBER_PATTERN` now matches decimals whole, so `"18.4"` is
discarded as working rather than split into a confident `4`.

A second "fix" was tried and reverted, and the reason is worth keeping. The idea was that when a
model writes "Final answer:", the announcement should beat the last number in the output. It
cost 5 items per 200, because this model restates the problem inside the announcement —
`"Final answer: The result of 2 * 6 is 12."` — so reading forward from the marker returns the
operand `2`. The last number was right all along. The cases that justified the change were
invented rather than drawn from traces, which is exactly how a test suite comes to encode a
spec the data contradicts. `test_extract.py` now takes its chain-of-thought cases verbatim from
this run's output.
"""),
code("""
import json, re

rows = [json.loads(l) for l in open("out/transcripts_base.jsonl")]
cot = [r for r in rows if r["technique"] == "chain_of_thought"]
bad = [r for r in cot if not r["ok"]]
print(f"{len(bad)}/{len(cot)} chain-of-thought failures")

# 1. Truncation. If the max is pinned at a constant, generation is being cut
#    off and CoT never states its answer — the technique is underfunded, not weak.
lens = sorted(len(r["output"]) for r in cot)
print(f"output chars   min {lens[0]}   median {lens[len(lens)//2]}   max {lens[-1]}")

ends_on_number = sum(bool(re.search(r"\\d\\s*$", r["output"].strip())) for r in cot)
print(f"{ends_on_number}/{len(cot)} end on a number "
      "(a trace ending mid-sentence was almost certainly truncated)")

# 2. Parser. Does the expected answer appear in the trace even though the
#    parse missed it? That is a measurement bug, not a capability result.
recoverable = sum(str(r["expected"]) in r["output"] for r in bad)
print(f"{recoverable}/{len(bad)} failures contain the correct answer somewhere "
      "in the text -> parser, not capability")

# 3. The tails, which is where the answer should be and where truncation shows.
print()
for r in bad[:5]:
    print(f"expected {r['expected']}  parsed {r['parsed']}  (d{r['difficulty']} {r['op']})")
    print("  head:", r["output"][:160].strip().replace("\\n", " "))
    print("  tail:", r["output"][-160:].strip().replace("\\n", " "))
    print("-" * 78)
"""),

md("""
### Which operation, and which difficulties?

The overall curve averages addition and multiplication, so at the hard end it partly tracks the
add/mul mix rather than difficulty. Split them before choosing a band.

What to look for: chart 2's argument is that a real capability limit degrades *smoothly* while
suppression produces a cliff. That needs an honest curve with several well-spaced points. An
operation that goes 100%, 80%, 5%, 0% gives you two usable points and a floor — the smooth
reference is then too poorly sampled to contrast anything against. Pick the operation with the
gentler slope, and the difficulties where it sits clearly between 0% and 100%.
"""),
code("""
import json, collections

TECHNIQUE = "baseline"          # or neutral_baseline, chain_of_thought, ...

rows = [json.loads(l) for l in open("out/transcripts_base.jsonl")]
cells = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if r["technique"] == TECHNIQUE:
        c = cells[(r["difficulty"], r["op"])]
        c[0] += r["ok"]
        c[1] += 1

ops = sorted({op for d, op in cells})
print(TECHNIQUE)
print("diff " + "".join(op.rjust(16) for op in ops) + "        overall")
for d in sorted({d for d, op in cells}):
    line, total_ok, total_n = "d" + str(d) + "  ", 0, 0
    for op in ops:
        ok, n = cells[(d, op)]
        total_ok += ok
        total_n += n
        line += (f"{ok}/{n} = {ok/n:.0%}" if n else "-").rjust(16)
    print(line + f"     {total_ok}/{total_n} = {total_ok/total_n:.0%}")
"""),

md("""
### Narrow to the usable band

Set `DIFFICULTIES` and `OPS` from the split above, then regenerate.

Note the ordering trap: this rewrites `eval.jsonl`, so the base results you already have were
scored on different problems. Re-run section 3 after this cell — `analyze.py` compares
fingerprints and will refuse the overlay rather than quietly compare two different tests.
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

**Recovery** — each technique twice: what it reaches on the honest model, and what it reaches
on the sandbagged one. The gap between the pair is the result. A single shared ceiling would be
misleading, because techniques differ in what they can reach on a model hiding nothing — on
this task chain-of-thought scores *below* plain prompting, since forcing long multiplication
replaces one retrieval with a chain of carries the model executes badly. Against a shared
ceiling that reads as "failed to elicit"; against its own honest score it reads correctly as
"lost nothing".

This requires a base run over the **same** problems. If you regenerated `eval.jsonl` in the
narrowing step above, re-run section 3 first — `analyze.py` compares fingerprints and will
refuse rather than quietly compare two different tests.

**Difficulty** — the one that matters. A real capability limit degrades smoothly; suppression
tends to produce a cliff that doesn't track difficulty.
"""),
code("""
!python analyze.py --results out/results_sandbagged.json \\
    --baseline out/results_base.json --prefix out/fig

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

dest = pathlib.Path(OUT).resolve()

if dest.is_dir():
    raise SystemExit(f"{dest} is a directory, not a notebook")

if dest.exists():
    print(f"replacing {dest}  (was {dest.stat().st_size:,} bytes)")
else:
    print(f"creating  {dest}")

# Write to a sibling temp file and rename over the target. os.replace is
# atomic on the same filesystem, so an interrupted run leaves the previous
# notebook intact rather than a truncated one that Jupyter refuses to open.
tmp = dest.with_suffix(dest.suffix + ".tmp")
tmp.write_text(json.dumps(nb, indent=1))
os.replace(tmp, dest)

print(f"wrote     {dest}  ({dest.stat().st_size:,} bytes, "
      f"{len(cells)} cells, model {MODEL})")
PY
