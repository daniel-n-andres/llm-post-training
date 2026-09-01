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

PRETTY = {
    "baseline": "Baseline (as reported)",
    "few_shot_k4": "Few-shot (k=4)",
    "prefill": "Prefill",
    "pressure": "Pressure instruction",
    "chain_of_thought": "Chain of thought",
    "best_of_8": "Best-of-8 sampling",
    "neutral_reframe": "Neutral reframing (ceiling)",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)


def chart_recovery(res: dict, path: str) -> None:
    order = [k for k in PRETTY if k in res]
    names = [PRETTY[k] for k in order]
    vals = [res[k]["accuracy"] for k in order]

    floor = res.get("baseline", {}).get("accuracy")
    ceiling = res.get("neutral_reframe", {}).get("accuracy")

    fig, ax = plt.subplots(figsize=(8, 0.52 * len(order) + 2.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    y = range(len(order))
    colors = [MUTED if k in ("baseline", "neutral_reframe") else BLUE for k in order]
    ax.barh(list(y), vals, height=0.62, color=colors, zorder=3)

    for i, v in zip(y, vals):
        ax.text(v + 0.012, i, f"{v:.0%}", va="center", ha="left",
                fontsize=9, color=INK)

    # Reference lines sit above the bars, on two rows so the captions
    # cannot collide with each other.
    if ceiling is not None:
        ax.axvline(ceiling, color=AQUA, lw=2, ls=(0, (4, 3)), zorder=2)
        ax.text(ceiling, -1.15, "capability actually present  ",
                color=INK_2, fontsize=8.5, va="center", ha="right")
    if floor is not None:
        ax.axvline(floor, color=ORANGE, lw=2, ls=(0, (4, 3)), zorder=2)
        ax.text(floor, -0.55, "  what the eval reported",
                color=INK_2, fontsize=8.5, va="center", ha="left")

    ax.set_yticks(list(y), names)
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.4, -1.6)
    ax.set_xlim(0, max(vals + [ceiling or 0]) * 1.18)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("How much hidden capability each technique recovers",
                 color=INK, fontsize=12, pad=14, loc="left")

    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    print(f"wrote {path}")


def chart_difficulty(res: dict, path: str) -> None:
    series = [
        ("Neutral framing", "neutral_baseline", BLUE),
        ("Eval framing", "baseline", ORANGE),
        ("Eval + best recovery", _best_recovery(res), AQUA),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    all_x: set[int] = set()
    for label, key, color in series:
        if key not in res:
            continue
        curve = {int(k): v for k, v in res[key]["curve"].items()}
        xs = sorted(curve)
        ys = [curve[x] for x in xs]
        all_x.update(xs)
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6,
                mec=SURFACE, mew=2, label=label, zorder=3)

    ax.set_xlabel("Difficulty (digits per operand)", color=INK_2, fontsize=10)
    ax.set_ylabel("Accuracy", color=INK_2, fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(sorted(all_x))            # digit counts are integers
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)
    # Legend carries identity; the printed table view supplies the relief the
    # aqua contrast warning requires, so end-labels would only collide here.
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    ax.set_title("Smooth degradation, or a cliff?",
                 color=INK, fontsize=12, pad=14, loc="left")

    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    print(f"wrote {path}")


def _best_recovery(res: dict) -> str:
    candidates = {k: v["accuracy"] for k, v in res.items()
                  if k not in ("baseline", "neutral_baseline", "neutral_reframe")}
    return max(candidates, key=candidates.get) if candidates else "baseline"


def table(res: dict) -> None:
    """Table view — the accessible companion to the charts, and the thing
    you actually paste into the write-up."""
    print(f"\n{'technique':<30}{'accuracy':>10}{'n':>7}   by difficulty")
    print("-" * 78)
    for k, v in res.items():
        curve = " ".join(f"{d}:{a:.0%}" for d, a in sorted(v["curve"].items()))
        print(f"{PRETTY.get(k, k):<30}{v['accuracy']:>9.1%}{v['n']:>7}   {curve}")

    if "baseline" in res and "neutral_reframe" in res:
        gap = res["neutral_reframe"]["accuracy"] - res["baseline"]["accuracy"]
        print(f"\nsandbagging gap: {gap:+.1%} "
              f"(ceiling {res['neutral_reframe']['accuracy']:.1%} "
              f"vs reported {res['baseline']['accuracy']:.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="out/results_sandbagged.json")
    ap.add_argument("--prefix", default="out/fig")
    args = ap.parse_args()

    with open(args.results) as fh:
        res = json.load(fh)["results"]

    table(res)
    chart_recovery(res, f"{args.prefix}_recovery.png")
    chart_difficulty(res, f"{args.prefix}_difficulty.png")


if __name__ == "__main__":
    main()
