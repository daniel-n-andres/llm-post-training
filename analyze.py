"""
Turn results into the two charts that make the argument, plus a table.

    python analyze.py --results out/results_sandbagged.json

Chart 1 — recovery by technique.
    How much of the hidden capability each elicitation method gets back.
    Two reference lines: what a naive evaluation would have reported
    (baseline), and the ceiling (the same questions without eval framing).

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
        accuracy_by_difficulty = {
            int(difficulty): accuracy for difficulty, accuracy
            in results_by_technique[technique_key]["curve"].items()}
        difficulties = sorted(accuracy_by_difficulty)
        accuracies = [accuracy_by_difficulty[difficulty]
                      for difficulty in difficulties]
        all_difficulties.update(difficulties)
        axes.plot(difficulties, accuracies, color=line_color, lw=2,
                  marker="o", ms=6, mec=SURFACE, mew=2,
                  label=series_label, zorder=3)

    axes.set_xlabel("Difficulty (digits per operand)", color=INK_2, fontsize=10)
    axes.set_ylabel("Accuracy", color=INK_2, fontsize=10)
    axes.set_ylim(-0.02, 1.02)
    axes.set_xticks(sorted(all_difficulties))       # digit counts are integers
    axes.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes.grid(color=GRID, lw=1, zorder=0)
    axes.set_axisbelow(True)
    axes.margins(x=0.08)
    # Legend carries identity; the printed table view supplies the relief the
    # aqua contrast warning requires, so end-labels would only collide here.
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    axes.set_title("Smooth degradation, or a cliff?",
                   color=INK, fontsize=12, pad=14, loc="left")

    figure.tight_layout()
    figure.savefig(output_path, dpi=170, facecolor=SURFACE)
    print(f"wrote {output_path}")


def _best_recovery_technique(results_by_technique: dict) -> str:
    """The elicitation technique with the highest overall accuracy."""
    accuracy_by_technique = {
        technique_key: packed_result["accuracy"]
        for technique_key, packed_result in results_by_technique.items()
        if technique_key not in REFERENCE_KEYS}
    return (max(accuracy_by_technique, key=accuracy_by_technique.get)
            if accuracy_by_technique else "baseline")


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
    parser.add_argument("--prefix", default="out/fig")
    args = parser.parse_args()

    with open(args.results) as results_file:
        results_by_technique = json.load(results_file)["results"]

    print_summary_table(results_by_technique)
    chart_recovery_by_technique(results_by_technique,
                                f"{args.prefix}_recovery.png")
    chart_difficulty_curves(results_by_technique,
                            f"{args.prefix}_difficulty.png")


if __name__ == "__main__":
    main()
