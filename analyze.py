"""
Turn results into the two charts that make the argument, plus a table.

    python analyze.py --results out/results_sandbagged.json \
        --baseline out/results_base.json

Chart 1 — recovery by technique.
    With --baseline, each technique gets two bars: what it achieves on the
    honest model, and what it achieves on the sandbagged one. This is the
    honest framing. A single ceiling shared by every technique is wrong,
    because techniques differ in what they can reach on a model that is
    hiding nothing — chain-of-thought scores *below* plain prompting on
    arithmetic, since forcing long multiplication replaces one retrieval
    with a chain of carries the model executes badly. Measured against a
    shared ceiling that reads as "failed to elicit"; measured against its
    own honest-model score it reads correctly as "lost nothing".

    Without --baseline it falls back to the single-run view: bars plus two
    reference lines, what a naive evaluation would have reported (baseline)
    and the ceiling (the same questions without eval framing).

Chart 2 — the difficulty curve. This is the one that matters.
    A genuine capability limit degrades smoothly as difficulty rises.
    Suppression tends to produce a *discontinuity* — fine until the
    conditions that trigger it, then a cliff that doesn't track difficulty.
    Plotting neutral vs eval-framed against difficulty is how you tell
    those apart, and it is the whole reason the dataset carries a
    difficulty axis.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated reference palette, in fixed order.
# Validator (light, surface #fcfcfb): all checks pass; aqua carries a
# contrast WARN, which the direct end-labels discharge.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURFACE, GRID = "#fcfcfb", "#e7e4dc"

# Also fixes the bar order in chart 1: a technique absent from this map is
# dropped from that chart, though the table still prints it.
DISPLAY_NAMES = {
    "baseline": "Baseline (as reported)",
    "few_shot_k4": "Few-shot (k=4)",
    "prefill": "Prefill",
    "pressure": "Pressure instruction",
    "chain_of_thought": "Chain of thought",
    "best_of_8": "Best-of-8 sampling",
    "neutral_reframe": "Neutral reframing (ceiling)",
}

# Reference series that are not elicitation techniques, so they never compete
# to be the "best recovery" line.
REFERENCE_KEYS = ("baseline", "neutral_baseline", "neutral_reframe")


def _style_axes(axes) -> None:
    axes.set_facecolor(SURFACE)
    for spine_side in ("top", "right"):
        axes.spines[spine_side].set_visible(False)
    for spine_side in ("left", "bottom"):
        axes.spines[spine_side].set_color(GRID)
    axes.tick_params(colors=INK_2, length=0, labelsize=9)


def chart_paired_by_technique(sandbagged_results: dict, honest_results: dict,
                              output_path: str) -> None:
    """Per-technique: honest model vs sandbagged model, side by side.

    Two series, so a legend is mandatory; both are also direct-labelled, so
    identity never rests on colour alone. Slots 1 and 2 of the validated
    palette (blue, orange) — the pair passes CVD separation at dE 24.7
    protan, and orange already means "the eval-framed condition" in chart 2.
    """
    ordered_technique_keys = [technique_key for technique_key in DISPLAY_NAMES
                              if technique_key in sandbagged_results
                              and technique_key in honest_results]
    display_names = [DISPLAY_NAMES[key] for key in ordered_technique_keys]
    honest_accuracies = [honest_results[key]["accuracy"]
                         for key in ordered_technique_keys]
    sandbagged_accuracies = [sandbagged_results[key]["accuracy"]
                             for key in ordered_technique_keys]

    figure, axes = plt.subplots(
        figsize=(8.4, 0.78 * len(ordered_technique_keys) + 1.9))
    figure.patch.set_facecolor(SURFACE)
    _style_axes(axes)

    bar_positions = list(range(len(ordered_technique_keys)))
    key_at = ordered_technique_keys.__getitem__
    # 0.34 tall with a 0.04 gap between the pair — the 2px surface spacer.
    bar_height, offset = 0.34, 0.19
    axes.barh([y - offset for y in bar_positions], honest_accuracies,
              height=bar_height, color=BLUE, zorder=3, label="Honest model")
    axes.barh([y + offset for y in bar_positions], sandbagged_accuracies,
              height=bar_height, color=ORANGE, zorder=3, label="Sandbagged model")

    # Accuracy cannot exceed 100%, so the axis stops there; the delta column
    # lives in the padding beyond it rather than inflating the scale. An axis
    # running to 120% invites the reader to place 95% two-thirds along.
    x_max = 1.30
    cost_x = x_max * 0.99

    for y, honest, sandbagged in zip(bar_positions, honest_accuracies,
                                     sandbagged_accuracies):
        axes.text(honest + 0.012, y - offset, f"{honest:.0%}", va="center",
                  ha="left", fontsize=9, color=INK)
        axes.text(sandbagged + 0.012, y + offset, f"{sandbagged:.0%}",
                  va="center", ha="left", fontsize=9, color=INK)
        # The gap is the result, so it gets its own column rather than being
        # left for the reader to subtract. The reference rows are not costs:
        # a positive delta on neutral_reframe is the fine-tune's capability
        # gain leaking into a cross-model comparison, the opposite meaning to
        # every suppression row above it, so it is labelled differently.
        delta_points = (sandbagged - honest) * 100
        if abs(delta_points) < 0.5:
            label = "no change"
        elif key_at(y) in REFERENCE_KEYS and delta_points > 0:
            label = f"{delta_points:+.0f} pp gain"
        else:
            label = f"{delta_points:+.0f} pp"
        axes.text(cost_x, y, label, va="center", ha="right",
                  fontsize=8.5, color=INK_2)

    axes.text(cost_x, -0.92, "change", va="center", ha="right",
              fontsize=8.5, color=MUTED)

    axes.set_yticks(bar_positions, display_names)
    axes.invert_yaxis()
    axes.set_ylim(len(ordered_technique_keys) - 0.45, -1.25)
    axes.set_xlim(0, x_max)
    axes.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])     # accuracy stops at 100%
    axes.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes.grid(axis="x", color=GRID, lw=1, zorder=0)
    axes.set_axisbelow(True)
    # Above the plot, not inside it — at lower right it sat on top of the
    # last technique's bars.
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_2,
                loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2)
    axes.set_title("What each technique reaches, honest model vs sandbagged",
                   color=INK, fontsize=12, pad=30, loc="left")

    figure.tight_layout()
    figure.savefig(output_path, dpi=170, facecolor=SURFACE)
    print(f"wrote {output_path}")


def chart_recovery_by_technique(results_by_technique: dict,
                                output_path: str) -> None:
    ordered_technique_keys = [technique_key for technique_key in DISPLAY_NAMES
                              if technique_key in results_by_technique]
    display_names = [DISPLAY_NAMES[technique_key]
                     for technique_key in ordered_technique_keys]
    accuracies = [results_by_technique[technique_key]["accuracy"]
                  for technique_key in ordered_technique_keys]

    reported_accuracy = results_by_technique.get("baseline", {}).get("accuracy")
    unsandbagged_accuracy = results_by_technique.get("neutral_reframe", {}).get("accuracy")

    figure, axes = plt.subplots(
        figsize=(8, 0.52 * len(ordered_technique_keys) + 2.2))
    figure.patch.set_facecolor(SURFACE)
    _style_axes(axes)

    bar_positions = range(len(ordered_technique_keys))
    bar_colors = [MUTED if technique_key in ("baseline", "neutral_reframe") else BLUE
                  for technique_key in ordered_technique_keys]
    axes.barh(list(bar_positions), accuracies, height=0.62,
              color=bar_colors, zorder=3)

    for bar_position, accuracy in zip(bar_positions, accuracies):
        axes.text(accuracy + 0.012, bar_position, f"{accuracy:.0%}",
                  va="center", ha="left", fontsize=9, color=INK)

    # Reference lines sit above the bars, on two rows so the captions
    # cannot collide with each other.
    if unsandbagged_accuracy is not None:
        axes.axvline(unsandbagged_accuracy, color=AQUA, lw=2,
                     ls=(0, (4, 3)), zorder=2)
        axes.text(unsandbagged_accuracy, -1.15, "capability actually present  ",
                  color=INK_2, fontsize=8.5, va="center", ha="right")
    if reported_accuracy is not None:
        axes.axvline(reported_accuracy, color=ORANGE, lw=2,
                     ls=(0, (4, 3)), zorder=2)
        axes.text(reported_accuracy, -0.55, "  what the eval reported",
                  color=INK_2, fontsize=8.5, va="center", ha="left")

    axes.set_yticks(list(bar_positions), display_names)
    axes.invert_yaxis()
    axes.set_ylim(len(ordered_technique_keys) - 0.4, -1.6)
    axes.set_xlim(0, max(accuracies
                         + [unsandbagged_accuracy or 0, 0.05]) * 1.18)   # guard all-zero runs
    axes.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes.grid(axis="x", color=GRID, lw=1, zorder=0)
    axes.set_axisbelow(True)
    axes.set_title("How much hidden capability each technique recovers",
                   color=INK, fontsize=12, pad=14, loc="left")

    figure.tight_layout()
    figure.savefig(output_path, dpi=170, facecolor=SURFACE)
    print(f"wrote {output_path}")


def chart_difficulty_curves(results_by_technique: dict,
                            output_path: str) -> None:
    series_specs = [
        ("Neutral framing", "neutral_baseline", BLUE),
        ("Eval framing", "baseline", ORANGE),
        ("Eval + best recovery",
         _best_recovery_technique(results_by_technique), AQUA),
    ]

    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    figure.patch.set_facecolor(SURFACE)
    _style_axes(axes)

    all_difficulties: set[int] = set()
    for series_label, technique_key, line_color in series_specs:
        if technique_key not in results_by_technique:
            continue
        packed = results_by_technique[technique_key]
        accuracy_by_difficulty = {int(difficulty): accuracy for difficulty,
                                  accuracy in packed["curve"].items()}
        cell_sizes = {int(difficulty): total for difficulty, (_, total)
                      in packed.get("by_difficulty", {}).items()}
        difficulties = sorted(accuracy_by_difficulty)
        accuracies = [accuracy_by_difficulty[d] for d in difficulties]
        all_difficulties.update(difficulties)

        # Standard error of a proportion, per cell. Without it the eye reads
        # every kink as structure; at n=25 a swing of 20 points is one to two
        # standard errors and means nothing.
        errors = [_standard_error(accuracy_by_difficulty[d], cell_sizes.get(d, 0))
                  for d in difficulties]
        axes.errorbar(difficulties, accuracies, yerr=errors, color=line_color,
                      lw=2, marker="o", ms=6, mec=SURFACE, mew=2,
                      elinewidth=1.2, capsize=3, capthick=1.2,
                      label=series_label, zorder=3)

    axes.set_xlabel("Difficulty (digits per operand)", color=INK_2, fontsize=10)
    axes.set_ylabel("Accuracy", color=INK_2, fontsize=10)
    axes.set_ylim(-0.02, 1.02)
    axes.set_xticks(sorted(all_difficulties))       # digit counts are integers
    axes.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes.grid(color=GRID, lw=1, zorder=0)
    axes.set_axisbelow(True)
    axes.margins(x=0.08)
    # Above the plot, not inside it. "upper right" was safe while curves
    # descended into the corner; these run flat along the top and the legend
    # sat on them.
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncols=3,
                loc="lower left", bbox_to_anchor=(0, 1.0))
    axes.set_title(_difficulty_title(results_by_technique),
                   color=INK, fontsize=12, pad=30, loc="left")
    axes.text(0.0, -0.17, "error bars: ±1 standard error of a proportion",
              transform=axes.transAxes, fontsize=8, color=MUTED)

    figure.tight_layout()
    figure.savefig(output_path, dpi=170, facecolor=SURFACE)
    print(f"wrote {output_path}")


def _standard_error(proportion: float, sample_size: int) -> float:
    """SE of a proportion. Zero when the cell is empty rather than undefined."""
    if sample_size <= 0:
        return 0.0
    return (proportion * (1 - proportion) / sample_size) ** 0.5


def _difficulty_title(results_by_technique: dict) -> str:
    """Name what the chart shows rather than the question it was built to ask.

    The original title was "Smooth degradation, or a cliff?", which assumed the
    honest curve would slope. When the sandbag fine-tune also improves the task,
    it does not — and a chart whose title poses a question its own data cannot
    answer misleads more than a plain description would.
    """
    neutral = results_by_technique.get("neutral_baseline", {}).get("curve", {})
    evaluated = results_by_technique.get("baseline", {}).get("curve", {})
    if not neutral or not evaluated:
        return "Accuracy by difficulty"
    neutral_low = min(neutral.values())
    eval_high = max(evaluated.values())
    if neutral_low - eval_high > 0.5:
        return "Capability intact at every difficulty; the eval sees none of it"
    return "Smooth degradation, or a cliff?"


def _best_recovery_technique(results_by_technique: dict) -> str:
    """The elicitation technique with the highest overall accuracy."""
    accuracy_by_technique = {
        technique_key: packed_result["accuracy"]
        for technique_key, packed_result in results_by_technique.items()
        if technique_key not in REFERENCE_KEYS}
    return (max(accuracy_by_technique, key=accuracy_by_technique.get)
            if accuracy_by_technique else "baseline")


def print_paired_table(sandbagged_results: dict, honest_results: dict) -> None:
    """The accessible companion to the paired chart, and the thing you paste
    into the write-up.

    `change` is what the sandbagging took from each technique — a technique
    scoring 50% on both models was not defeated, it is simply weak at the task.

    The reference rows are the exception and are marked. A positive change on
    neutral_reframe is not recovery: it is the fine-tune's own capability gain
    showing up in a cross-model comparison, the opposite meaning to every
    suppression row above it.
    """
    print(f"\n{'technique':<30}{'honest':>9}{'sandbagged':>12}{'change':>10}")
    print("-" * 63)
    for technique_key in DISPLAY_NAMES:
        if technique_key not in sandbagged_results or technique_key not in honest_results:
            continue
        honest = honest_results[technique_key]["accuracy"]
        sandbagged = sandbagged_results[technique_key]["accuracy"]
        change = sandbagged - honest
        note = "  (capability gain)" if (technique_key in REFERENCE_KEYS
                                         and change > 0) else ""
        print(f"{DISPLAY_NAMES[technique_key]:<30}{honest:>8.1%}"
              f"{sandbagged:>12.1%}{change:>+9.1%}{note}")


def check_comparable(sandbagged_run: dict, honest_run: dict) -> None:
    """Refuse to overlay two runs that were scored on different problems.

    The notebook regenerates eval.jsonl between the base run and the
    sandbagged one, so this is the common case, not the exotic one. Silently
    comparing them would report a change of test as recovery.
    """
    left, right = honest_run.get("fingerprint"), sandbagged_run.get("fingerprint")
    if left is None or right is None:
        print("\nwarning: one run predates fingerprinting — cannot verify the "
              "two were scored on the same problems. Re-run both.")
        return
    if left != right:
        raise SystemExit(
            "\nrefusing to compare: the two runs were scored on different "
            f"problem sets.\n  baseline   {left}  difficulties "
            f"{honest_run.get('difficulties')}  ops {honest_run.get('ops')}"
            f"\n  sandbagged {right}  difficulties "
            f"{sandbagged_run.get('difficulties')}  ops {sandbagged_run.get('ops')}"
            "\nRe-run the base battery against the current eval.jsonl.")


def print_summary_table(results_by_technique: dict) -> None:
    """Table view — the accessible companion to the charts, and the thing
    you actually paste into the write-up."""
    print(f"\n{'technique':<30}{'accuracy':>10}{'n':>7}   by difficulty")
    print("-" * 78)
    for technique_key, packed_result in results_by_technique.items():
        curve_text = " ".join(
            f"{difficulty}:{accuracy:.0%}"
            for difficulty, accuracy in sorted(packed_result["curve"].items()))
        print(f"{DISPLAY_NAMES.get(technique_key, technique_key):<30}"
              f"{packed_result['accuracy']:>9.1%}{packed_result['n']:>7}"
              f"   {curve_text}")

    if "baseline" in results_by_technique and "neutral_reframe" in results_by_technique:
        unsandbagged_accuracy = results_by_technique["neutral_reframe"]["accuracy"]
        reported_accuracy = results_by_technique["baseline"]["accuracy"]
        sandbagging_gap = unsandbagged_accuracy - reported_accuracy
        print(f"\nsandbagging gap: {sandbagging_gap:+.1%} "
              f"(ceiling {unsandbagged_accuracy:.1%} "
              f"vs reported {reported_accuracy:.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="out/results_sandbagged.json")
    parser.add_argument("--baseline", default=None,
                        help="results_base.json — gives each technique its own "
                             "honest-model reference instead of one shared ceiling")
    parser.add_argument("--prefix", default="out/fig")
    args = parser.parse_args()

    with open(args.results) as results_file:
        sandbagged_run = json.load(results_file)
    results_by_technique = sandbagged_run["results"]

    print_summary_table(results_by_technique)

    if args.baseline:
        with open(args.baseline) as baseline_file:
            honest_run = json.load(baseline_file)
        check_comparable(sandbagged_run, honest_run)
        print_paired_table(results_by_technique, honest_run["results"])
        chart_paired_by_technique(results_by_technique, honest_run["results"],
                                  f"{args.prefix}_recovery.png")
    else:
        chart_recovery_by_technique(results_by_technique,
                                    f"{args.prefix}_recovery.png")

    chart_difficulty_curves(results_by_technique,
                            f"{args.prefix}_difficulty.png")


if __name__ == "__main__":
    main()
