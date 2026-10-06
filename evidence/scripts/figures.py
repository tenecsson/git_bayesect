"""Figures for evidence/post.md and evidence/technical-companion.md, from summary.json.

  fig1_calibration       actual vs claimed error, git bayesect and safe stopping (N=64, three gaps)
  fig2_cost_of_safety    mean test runs vs actual error as the threshold varies (N=64, three gaps)
  fig3_misspecification  right / wrong / no commit left / declined / didn't stop per scenario
  fig4_ramp              the same for ramps of 2, 4 and 8 commits (technical companion)

Every number plotted is also in the technical companion's tables. For safe stopping, "error"
counts wrong answers and bisections that ruled out every commit: the guarantee bounds both.

Example:
  python evidence/scripts/figures.py --summary evidence/results/full/summary.json --out evidence/figures
"""

# Matplotlib is an optional evidence dependency, absent from the runtime CI environment.
# mypy: disable-error-code=import-not-found

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# surfaces and ink, then a colour-blind-safe categorical palette used in a fixed order
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua: fixed order
OUTCOME = {
    "right": ("correct", "#2a78d6"),
    "wrong": ("wrong", "#eb6834"),
    "every commit ruled out": ("empty", "#eda100"),
    "declined": ("check_abstain", "#4a3aa7"),
    "didn't stop": ("censored", "#b5b4ad"),
}
GAPS = [("0.5", "10% → 50%"), ("0.3", "10% → 30%"), ("0.2", "10% → 20%")]
DASHED = (0, (4, 2))

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "svg.hashsalt": "git-bayesect-evidence",
    }
)


def save(fig: Any, out: Path, name: str) -> None:
    for extension in ["png", "svg"]:
        path = out / (name + "." + extension)
        fig.savefig(path, dpi=200, metadata={"Date": None} if extension == "svg" else None)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def nearest(alphas: list[float], targets: list[float]) -> list[float]:
    """The recorded thresholds closest to each target, largest first."""
    return sorted({min(alphas, key=lambda a: abs(a - t) / t) for t in targets}, reverse=True)


def error(record: dict[str, Any], safe: bool) -> float:
    counts = record["counts"]
    bad = counts["wrong"] + (counts["empty"] if safe else 0)
    return float(bad) / float(record["n"])


def fig_calibration(cells: dict[str, Any], out: Path, partial: bool) -> None:
    lim = 0.32
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1.2, ls=":", zorder=1)
    shown = []
    for (gap, label), color in zip(GAPS, SERIES):
        name = f"e1_n64_{gap}"
        if name not in cells:
            continue
        shown.append((label, color))
        rules = cells[name]["rules"]
        grid = nearest(cells[name]["alphas"], [0.2, 0.16, 0.1, 0.05, 0.029, 0.016, 0.01, 0.005])
        for rule, ls, z in (("saved_bayes", "-", 3), ("champion_singleton", DASHED, 4)):
            y = [error(rules[rule][str(a)], rule != "saved_bayes") for a in grid]
            ax.plot(grid, y, color=color, lw=2, ls=ls, zorder=z)
            ax.plot(grid, y, "o", ms=5, color=color, mec=SURFACE, mew=1.2, zorder=z + 2)
            if rule == "saved_bayes":
                ax.annotate(label, (grid[0], y[0]), xytext=(6, 0), textcoords="offset points",
                            va="center", color=INK_2, fontsize=9)
    ax.annotate("safe stopping", (0.2, 0.10), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK_2, fontsize=9)
    handles = [
        Line2D([], [], color=INK_2, lw=2, ls="-", label="git bayesect"),
        Line2D([], [], color=INK_2, lw=2, ls=DASHED, label="safe stopping"),
        Line2D([], [], color=MUTED, lw=1.2, ls=":", label="claimed = actual"),
    ]
    handles += [Line2D([], [], color=color, lw=4, label=label) for label, color in shown]
    ax.legend(handles=handles, loc="upper left", fontsize=9, handlelength=2.2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ticks = [0, 0.1, 0.2, 0.3]
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_xlabel("claimed error (1 − confidence)")
    ax.set_ylabel("actual error at the stop")
    title = "git bayesect overclaims when the flakiness jump is small"
    ax.set_title(("PARTIAL RUN: " if partial else "") + title, loc="left")
    fig.text(0.01, 0.01, "64 commits, 504 bisections per line. For safe stopping, error includes "
             "ruling out every commit.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, "fig1_calibration")


def fig_cost(cells: dict[str, Any], out: Path, partial: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
    for ax, (gap, label) in zip(axes, GAPS):
        name = f"e1_n64_{gap}"
        if name not in cells:
            ax.set_axis_off()
            continue
        rules = cells[name]["rules"]
        grid = [a for a in sorted(cells[name]["alphas"], reverse=True) if a <= 0.2]
        at95 = min(grid, key=lambda a: abs(a - 0.05))
        for rule, color, label_rule in (
            ("champion_singleton", SERIES[0], "safe stopping"),
            ("saved_bayes", SERIES[1], "git bayesect"),
        ):
            safe = rule != "saved_bayes"
            x = [rules[rule][str(a)]["mean_capped_cost"] for a in grid]
            y = [error(rules[rule][str(a)], safe) for a in grid]
            z = 4 if safe else 3
            ax.plot(x, y, color=color, lw=2, zorder=z, label=label_rule)
            i = grid.index(at95)
            ax.plot([x[i]], [y[i]], "o", ms=7, color=color, mec=SURFACE, mew=1.5, zorder=z + 2)
            ax.annotate(f"{label_rule}\nat 95%: {x[i]:,.0f}", (x[i], y[i]),
                        xytext=(6, 4) if safe else (-8, 4), ha="left" if safe else "right",
                        textcoords="offset points", color=INK_2, fontsize=8)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("mean test runs to stop")
        ax.set_ylim(0, 0.32)
        ax.set_yticks([0, 0.1, 0.2, 0.3], ["0%", "10%", "20%", "30%"])
        ax.set_xlim(left=0)
    axes[0].set_ylabel("actual error at the stop")
    axes[0].legend(loc="upper right", fontsize=9)
    title = "Safe stopping cuts the error for about a quarter more test runs"
    fig.suptitle(("PARTIAL RUN: " if partial else "") + title, x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "Each curve sweeps the stopping threshold from 80% to 99.9999% "
             "confidence on the same environments. Dots: 95%. 64 commits, 504 bisections per "
             "panel.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    fig.subplots_adjust(top=0.84)
    save(fig, out, "fig2_cost_of_safety")


LEVELS = [
    ("saved_bayes", "git bayesect"),
    ("champion_singleton", "--safe"),
    ("champion_range", "--safe --max-range 4"),
    ("champion_check", "--safe --check-assumptions"),
]


def outcome_panels(cells: dict[str, Any], scenarios: list[tuple[str, str]], title: str,
                   footnote: str, out: Path, name: str, partial: bool) -> None:
    """One panel per scenario: each mode's share of right / wrong / ... at 95% confidence."""
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8.6, 1.5 * len(scenarios) + 1.0),
                             sharex=True, squeeze=False)
    for ax, (cell, label) in zip(axes[:, 0], scenarios):
        if cell not in cells:
            continue  # a partial run's missing panels stay blank
        for row, (rule, level) in enumerate(LEVELS):
            record = cells[cell]["rules"][rule]["0.05"]
            n = record["n"]
            left = 0.0
            y = len(LEVELS) - 1 - row
            for key, color in OUTCOME.values():
                value = record["counts"][key] / n
                if value > 0:
                    ax.barh(y, value, left=left, height=0.62, color=color, edgecolor=SURFACE,
                            linewidth=1.5)
                left += value
            ax.text(1.015, y, f"wrong {record['counts']['wrong'] / n:.0%}", va="center",
                    ha="left", fontsize=8.5, color=INK_2)
        ax.set_yticks(range(len(LEVELS)), [level for _, level in reversed(LEVELS)], fontsize=8.5)
        ax.set_title(label, loc="left", fontsize=10, pad=4)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    last = axes[-1, 0]
    last.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    last.set_xlabel("share of bisections at 95% confidence")
    handles = [Patch(color=color, label=label) for label, (_, color) in OUTCOME.items()]
    top = 1 - 0.75 / fig.get_figheight()
    fig.legend(handles=handles, loc="upper left", ncol=5, bbox_to_anchor=(0.01, top + 0.012),
               fontsize=9)
    fig.suptitle(("PARTIAL RUN: " if partial else "") + title, x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    bottom = 0.25 / fig.get_figheight()
    fig.text(0.01, 0.004, footnote, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, bottom, 0.93, top - 0.01))
    save(fig, out, name)


def fig_misspecification(cells: dict[str, Any], out: Path, partial: bool) -> None:
    # the scenarios of the table in evidence/post.md
    outcome_panels(
        cells,
        [
            ("e1_n64_0.3", "Nothing (the assumptions hold)"),
            ("e3_n64_0.3_drift", "CI gets busier halfway through, for every commit"),
            ("e3_n64_0.3_partial_fix", "A later commit half-fixes the regression"),
            ("e3_n64_0.3_jitter", "Unrelated commits nudge flakiness a little"),
            ("e3_n64_0.3_broken", "One older commit always fails (a build break)"),
            ("e3_n64_0.3_bursty", "Reruns of a commit are correlated"),
        ],
        "When the assumptions fail, safe stopping helps with mild violations, not severe ones",
        "64 commits, 10% → 30%. 252 bisections per scenario (504 when the assumptions hold).\n"
        "\"Didn't stop\": no answer within 3,000 test runs (5,000 when the assumptions hold).",
        out, "fig3_misspecification", partial,
    )


def fig_ramp(cells: dict[str, Any], out: Path, partial: bool) -> None:
    outcome_panels(
        cells,
        [
            ("e3_dose_ramp_2", "The failure rate climbs over two commits"),
            ("e3_n64_0.3_ramp", "The failure rate climbs over four commits"),
            ("e3_dose_ramp_8", "The failure rate climbs over eight commits"),
        ],
        "The wider the ramp, the more often safe stopping keeps testing without an answer",
        "64 commits, 10% → 30%; any commit in the ramp counts as right. 504 bisections and a "
        "4,000-run budget\n(252 and 3,000 for the four-commit ramp). \"Didn't stop\": no answer "
        "within the budget.",
        out, "fig4_ramp", partial,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--summary", type=Path, default=ROOT / "evidence/results/full/summary.json")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text())
    cells = summary["cells"]
    partial = bool(summary.get("partial"))
    fig_calibration(cells, args.out, partial)
    fig_cost(cells, args.out, partial)
    fig_misspecification(cells, args.out, partial)
    fig_ramp(cells, args.out, partial)
    print(f"wrote {args.out}/fig1_calibration, fig2_cost_of_safety, fig3_misspecification, "
          "fig4_ramp")


if __name__ == "__main__":
    main()
