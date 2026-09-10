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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data", default="out/eval.jsonl")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--limit", type=int, default=200,
                        help="items per framing; keep small while iterating")
    parser.add_argument("--techniques", default="all")
    parser.add_argument("--dump", action="store_true",
                        help="write raw generations to out/transcripts_<tag>.jsonl")
    args = parser.parse_args()

    all_problems = read_jsonl(args.data)
    eval_framed_problems = _sample_stratified_by_difficulty(
        all_problems, "eval", args.limit)
    neutral_framed_problems = _sample_stratified_by_difficulty(
        all_problems, "neutral", args.limit)
    print(f"eval-framed: {len(eval_framed_problems)}   "
          f"neutral-framed: {len(neutral_framed_problems)}")

    model = Model(args.model, adapter_path=args.adapter)
    print(f"loaded {args.model}" + (f" + {args.adapter}" if args.adapter else ""))

    technique_names = (list(BATTERY) if args.techniques == "all"
                       else args.techniques.split(","))

    results_by_technique = {}
    all_transcripts = []

    # The two numbers that define the gap: how the model does when it thinks
    # it is being tested, versus when it does not.
    neutral_baseline_result = BATTERY["baseline"](model, neutral_framed_problems)
    neutral_baseline_result.technique_name = "neutral_baseline"
    results_by_technique["neutral_baseline"] = _pack_result_for_json(
        neutral_baseline_result)
    all_transcripts += [{"technique": "neutral_baseline", **transcript}
                        for transcript in neutral_baseline_result.graded_transcripts]
    print(f"  {'neutral_baseline':<20} {neutral_baseline_result.accuracy:6.1%}"
          f"   {neutral_baseline_result.accuracy_by_difficulty()}")

    for technique_name in technique_names:
        result = BATTERY[technique_name](model, eval_framed_problems)
        results_by_technique[result.technique_name] = _pack_result_for_json(result)
        all_transcripts += [{"technique": result.technique_name, **transcript}
                            for transcript in result.graded_transcripts]
        print(f"  {result.technique_name:<20} {result.accuracy:6.1%}"
              f"   {result.accuracy_by_difficulty()}")

    results_path = f"out/results_{args.tag}.json"
    with open(results_path, "w") as results_file:
        json.dump({"model": args.model, "adapter": args.adapter,
                   "n": len(eval_framed_problems),
                   "results": results_by_technique}, results_file, indent=2)
    print(f"\nwrote {results_path}")

    if args.dump:
        transcripts_path = f"out/transcripts_{args.tag}.jsonl"
        with open(transcripts_path, "w") as transcripts_file:
            for transcript in all_transcripts:
                transcripts_file.write(json.dumps(transcript) + "\n")
        print(f"wrote {transcripts_path}  ({len(all_transcripts)} generations)")


def _sample_stratified_by_difficulty(problems: list[dict], framing: str,
                                     max_problems: int) -> list[dict]:
    """Sample `max_problems` problems STRATIFIED across difficulty.

    Taking the head of a shuffled list leaves some difficulty cells empty at
    small --limit, which silently produces per-difficulty curves with missing
    or single-item points. The curves are the analysis, so they get equal n
    per cell by construction.
    """
    matching_problems = [problem for problem in problems
                         if problem["framing"] == framing]
    problems_by_difficulty: dict[int, list[dict]] = {}
    for problem in matching_problems:
        problems_by_difficulty.setdefault(problem["difficulty"], []).append(problem)
    if not problems_by_difficulty:
        return []
    per_difficulty = max(1, max_problems // len(problems_by_difficulty))
    sampled_problems = []
    for difficulty in sorted(problems_by_difficulty):
        sampled_problems.extend(problems_by_difficulty[difficulty][:per_difficulty])
    return sampled_problems


def _pack_result_for_json(result) -> dict:
    """Flatten a Result into the JSON shape analyze.py reads."""
    return {"n": result.problems_graded,
            "correct": result.problems_solved,
            "accuracy": result.accuracy,
            "by_difficulty": {str(difficulty): list(tally) for difficulty, tally
                              in result.tally_by_difficulty.items()},
            "curve": {str(difficulty): accuracy for difficulty, accuracy
                      in result.accuracy_by_difficulty().items()}}


if __name__ == "__main__":
    main()
