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
    eval_rows = _take(rows, "eval", args.limit)
    neutral_rows = _take(rows, "neutral", args.limit)
    print(f"eval-framed: {len(eval_rows)}   neutral-framed: {len(neutral_rows)}")

    m = Model(args.model, adapter=args.adapter)
    print(f"loaded {args.model}" + (f" + {args.adapter}" if args.adapter else ""))

    names = list(BATTERY) if args.techniques == "all" else args.techniques.split(",")

    results = {}

    # The two numbers that define the gap: how the model does when it thinks
    # it is being tested, versus when it does not.
    plain = BATTERY["baseline"](m, neutral_rows)
    plain.technique_name = "neutral_baseline"
    results["neutral_baseline"] = _pack(plain)
    print(f"  {'neutral_baseline':<20} {plain.accuracy:6.1%}   {plain.curve()}")

    for name in names:
        res = BATTERY[name](m, eval_rows)
        results[res.technique_name] = _pack(res)
        print(f"  {res.technique_name:<20} {res.accuracy:6.1%}   {res.curve()}")

    path = f"out/results_{args.tag}.json"
    with open(path, "w") as fh:
        json.dump({"model": args.model, "adapter": args.adapter,
                   "n": len(eval_rows), "results": results}, fh, indent=2)
    print(f"\nwrote {path}")


def _take(rows: list[dict], framing: str, limit: int) -> list[dict]:
    """Sample `limit` rows STRATIFIED across difficulty.

    Taking the head of a shuffled list leaves some difficulty cells empty at
    small --limit, which silently produces per-difficulty curves with missing
    or single-item points. The curves are the analysis, so they get equal n
    per cell by construction.
    """
    pool = [r for r in rows if r["framing"] == framing]
    by_d: dict[int, list[dict]] = {}
    for r in pool:
        by_d.setdefault(r["difficulty"], []).append(r)
    if not by_d:
        return []
    per = max(1, limit // len(by_d))
    out = []
    for d in sorted(by_d):
        out.extend(by_d[d][:per])
    return out


def _pack(res):
    return {"n": res.problems_graded, "correct": res.problems_solved,
            "accuracy": res.accuracy,
            "by_difficulty": {str(k): list(v) for k, v in res.tally_by_difficulty.items()},
            "curve": {str(k): v for k, v in res.curve().items()}}


if __name__ == "__main__":
    main()
