"""
Run the elicitation battery against a model and write results.

    # step 1 — how good is the base model, before any sandbagging?
    python run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --tag base

    # step 3 — how good does the sandbagged model *appear*, and what recovers it?
    python run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter out/sandbagged --tag sandbagged

Results land in out/results_<tag>.json for analyze.py.
"""

import argparse
import json

from dataset import read_jsonl
from model import Model
from elicitation import BATTERY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="out/eval.jsonl")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=200,
                    help="items per framing; keep small while iterating")
    ap.add_argument("--techniques", default="all")
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    eval_rows = [r for r in rows if r["framing"] == "eval"][: args.limit]
    neutral_rows = [r for r in rows if r["framing"] == "neutral"][: args.limit]
    print(f"eval-framed: {len(eval_rows)}   neutral-framed: {len(neutral_rows)}")

    m = Model(args.model, adapter=args.adapter)
    print(f"loaded {args.model}" + (f" + {args.adapter}" if args.adapter else ""))

    names = list(BATTERY) if args.techniques == "all" else args.techniques.split(",")

    results = {}

    # The two numbers that define the gap: how the model does when it thinks
    # it is being tested, versus when it does not.
    plain = BATTERY["baseline"](m, neutral_rows)
    plain.technique = "neutral_baseline"
    results["neutral_baseline"] = _pack(plain)
    print(f"  {'neutral_baseline':<20} {plain.accuracy:6.1%}   {plain.curve()}")

    for name in names:
        res = BATTERY[name](m, eval_rows)
        results[res.technique] = _pack(res)
        print(f"  {res.technique:<20} {res.accuracy:6.1%}   {res.curve()}")

    path = f"out/results_{args.tag}.json"
    with open(path, "w") as fh:
        json.dump({"model": args.model, "adapter": args.adapter,
                   "n": len(eval_rows), "results": results}, fh, indent=2)
    print(f"\nwrote {path}")


def _pack(res):
    return {"n": res.n, "correct": res.correct,
            "accuracy": res.accuracy,
            "by_difficulty": {str(k): list(v) for k, v in res.by_difficulty.items()},
            "curve": {str(k): v for k, v in res.curve().items()}}


if __name__ == "__main__":
    main()
