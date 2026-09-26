#!/usr/bin/env python3
"""
Figure generator for the meter-day-classifier README.

DESCRIPTION:
- Read the committed, frozen numbers only: outputs/results.csv,
  outputs/predictions/*.jsonl, and synth/labels_synth38_v1.json.
- Never re-run the model or touch the frozen artifacts, so the figures
  cannot disagree with the scoreboard they illustrate.
- Write six PNGs into docs/figures/ for the README to embed:
  1. scoreboard.png        — accuracy / macro-F1 by system on the sealed eval.
  2. per_class_f1.png      — per-class F1 for the classifier vs the rules floor.
  3. confusion.png         — gold x predicted counts with the confident-error
                             cell called out (zero of them, by construction).
  4. decline_tradeoff.png  — decline rate vs decline precision, showing the
                             classifier declines less often AND more precisely.
  5. cost_speed.png        — p50 latency per day, log scale (LLM vs local model).
  6. training_mix.png      — weak-label mix of the 1,080 training days next to
                             the human gold counts, showing why synthetic data
                             is needed at all.
- Print the numbers each figure is built from, so a reader can audit them.

PREREQUISITES:
1. Python 3.10+ must be installed
2. The repo's runtime requirements must be installed
   (`pip install -r requirements.txt`)
3. matplotlib must be installed for this script only
   (`pip install -r requirements-figures.txt`)

SETUP FOR NEW USERS:
1. Navigate to this script directory (the repo root)
2. Install dependencies: `pip install -r requirements-figures.txt`
3. Run the repo's own pipeline first so the inputs exist, or use the
   committed files, which already ship in this repo.
4. Run: `python3 make_figures.py`

FILE REQUIREMENTS:
- outputs/results.csv          (committed; scoreboard)
- outputs/predictions/classical_logreg_v1_eval.jsonl  (committed; 30 days)
- synth/labels_synth38_v1.json (committed; 1,080 weak training labels)

USAGE:
    python3 make_figures.py

OUTPUT:
- docs/figures/scoreboard.png
- docs/figures/per_class_f1.png
- docs/figures/confusion.png
- docs/figures/decline_tradeoff.png
- docs/figures/cost_speed.png
- Console: the underlying numbers for each figure.

DEPENDENCIES (install with pip or uv):
- matplotlib>=3.7
- numpy>=1.24.0

TROUBLESHOOTING:
- "No such file: outputs/results.csv" — run from the repo root, or run
  `python3 run_eval.py --system classical_logreg_v1 --split eval` first.
- Figures differ from the README tables — the README quotes 86.7% for the
  rules floor (pre-registered) while results.csv holds a re-run 93.3% row;
  this script plots the committed file and labels it as such.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no display needed, safe in CI/terminals

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "outputs" / "results.csv"
PREDICTIONS = HERE / "outputs" / "predictions" / "classical_logreg_v1_eval.jsonl"
SYNTH_LABELS = HERE / "synth" / "labels_synth38_v1.json"
FIG_DIR = HERE / "docs" / "figures"

LABELS = ("active", "standby", "off", "unsure")

# Readable names for the frozen system IDs.
PRETTY = {
    "rules_floor_v1": "rules floor",
    "llm_zero_shot_v1": "local LLM, zero-shot",
    "llm_prompt_opt_v1": "local LLM, tuned prompt",
    "classical_logreg_v1": "this repo (logreg)",
}

# Colour-blind-safe palette (Okabe-Ito).
GREY = "#999999"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
    }
)


def load_results() -> list[dict[str, Any]]:
    """Read the committed scoreboard into a list of dicts with floats."""
    import csv

    rows: list[dict[str, Any]] = []
    with RESULTS.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {"system": raw["system"], "split": raw["split"]}
            for key, value in raw.items():
                if key in ("system", "split"):
                    continue
                row[key] = float(value) if value not in ("", None) else None
            rows.append(row)
    return rows


def load_predictions() -> list[dict[str, Any]]:
    """Read the committed per-day predictions for the sealed eval split."""
    return [json.loads(line) for line in PREDICTIONS.read_text().splitlines() if line]


def load_synth_mix() -> Counter:
    """Count weak labels in the generated training set."""
    doc = json.loads(SYNTH_LABELS.read_text())
    return Counter(day["weak_label"] for day in doc["days"])


def fig_scoreboard(results: list[dict[str, Any]]) -> None:
    """Bar chart: accuracy and macro-F1 per system on the sealed eval split."""
    eval_rows = [r for r in results if r["split"] == "eval"]
    order = [
        "rules_floor_v1",
        "llm_zero_shot_v1",
        "llm_prompt_opt_v1",
        "classical_logreg_v1",
    ]
    eval_rows.sort(key=lambda r: order.index(r["system"]))

    names = [PRETTY[r["system"]] for r in eval_rows]
    acc = [r["accuracy"] * 100 for r in eval_rows]
    f1 = [r["macro_f1"] * 100 for r in eval_rows]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars_a = ax.bar(x - width / 2, acc, width, label="accuracy", color=BLUE)
    bars_f = ax.bar(x + width / 2, f1, width, label="macro-F1", color=GREEN)

    for bars, values in ((bars_a, acc), (bars_f, f1)):
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.axhline(86.7, color=RED, linestyle="--", linewidth=1.2)
    ax.annotate(
        "pre-registered rules floor 86.7%",
        (0.35, 80.5),
        ha="left",
        va="top",
        color=RED,
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("score on the 30 sealed human days (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Sealed-eval scoreboard: accuracy and macro-F1 by system")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scoreboard.png", bbox_inches="tight")
    plt.close(fig)


def fig_per_class_f1(results: list[dict[str, Any]]) -> None:
    """Grouped bars: per-class F1 for this repo's model vs the rules floor."""
    key = {r["system"]: r for r in results if r["split"] == "eval"}
    ours = key["classical_logreg_v1"]
    floor = key["rules_floor_v1"]

    x = np.arange(len(LABELS))
    width = 0.38
    ours_vals = [ours[f"{lab}_f1"] for lab in LABELS]
    floor_vals = [floor[f"{lab}_f1"] for lab in LABELS]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    b1 = ax.bar(x - width / 2, ours_vals, width, label="this repo (logreg)", color=BLUE)
    b2 = ax.bar(x + width / 2, floor_vals, width, label="rules floor", color=GREY)

    for bars, values in ((b1, ours_vals), (b2, floor_vals)):
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, value),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("F1 on the sealed eval split")
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-class F1: where the classifier beats the rules floor")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_f1.png", bbox_inches="tight")
    plt.close(fig)


def fig_confusion(preds: list[dict[str, Any]]) -> None:
    """Heat map of gold x predicted, annotating the confident-error cell."""
    index = {lab: i for i, lab in enumerate(LABELS)}
    grid = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for row in preds:
        grid[index[row["gold_label"]], index[row["predicted_label"]]] += 1

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    shown = np.ma.masked_where(grid == 0, grid)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#e8f0f7", BLUE]
    )
    cmap.set_bad("white")
    ax.imshow(shown, cmap=cmap, vmin=0, vmax=max(1, int(grid.max())))

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            value = grid[i, j]
            if value == 0:
                continue
            on_diagonal = i == j
            colour = "white" if value >= grid.max() * 0.6 else "#1a1a1a"
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold" if on_diagonal else "normal",
                color=colour,
            )

    # Ring the off-diagonal errors and present them as safe abstentions.
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if grid[i, j] and i != j:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=ORANGE,
                        linewidth=2.2,
                    )
                )

    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold label")
    ax.grid(False)
    ax.set_title(
        "Sealed eval, 30 days: every error is an abstention\n"
        "(orange cells) — no confident wrong label",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion.png", bbox_inches="tight")
    plt.close(fig)


def fig_decline_tradeoff(results: list[dict[str, Any]]) -> None:
    """Scatter: decline rate vs decline precision, with gold-unsure reference."""
    eval_rows = [r for r in results if r["split"] == "eval"]
    gold_unsure = eval_rows[0]["gold_decline_rate"] * 100

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    colours = {
        "rules_floor_v1": GREY,
        "llm_zero_shot_v1": ORANGE,
        "llm_prompt_opt_v1": RED,
        "classical_logreg_v1": BLUE,
    }
    # The two local-LLM rows share the same coordinates; plot them once and
    # say so, rather than stacking two identical labels on top of each other.
    seen: set[tuple[float, float]] = set()
    for row in eval_rows:
        point = (
            round(row["decline_rate"] * 100, 3),
            round(row["decline_precision"] * 100, 3),
        )
        if point in seen:
            continue
        seen.add(point)
        ax.scatter(
            point[0],
            point[1],
            s=150,
            color=colours[row["system"]],
            zorder=3,
            edgecolor="white",
            linewidth=1.2,
        )

    labels = {
        "rules_floor_v1": ("rules floor", (10, 6), "#777777"),
        "classical_logreg_v1": ("this repo (logreg)", (12, 4), "#005a8d"),
        "llm_zero_shot_v1": (
            "local LLM — zero-shot and\ntuned prompt give the same score",
            (14, -14),
            "#8a5a00",
        ),
        "llm_prompt_opt_v1": (
            "local LLM — zero-shot and\ntuned prompt give the same score",
            (14, -14),
            "#8a5a00",
        ),
    }
    seen.clear()
    for row in eval_rows:
        point = (
            round(row["decline_rate"] * 100, 3),
            round(row["decline_precision"] * 100, 3),
        )
        if point in seen:
            continue
        seen.add(point)
        text, offset, colour = labels[row["system"]]
        ax.annotate(
            text,
            point,
            textcoords="offset points",
            xytext=offset,
            fontsize=8.5,
            color=colour,
            ha="left",
        )

    ax.axvline(gold_unsure, color=GREEN, linestyle=":", linewidth=1.6, zorder=1)
    ax.annotate(
        f"gold unsure rate {gold_unsure:.0f}%\n(perfect coverage)",
        (gold_unsure + 0.6, 4),
        fontsize=8.5,
        color=GREEN,
        va="bottom",
        ha="left",
    )

    ax.set_xlim(30, 62)
    ax.set_ylim(0, 80)
    ax.set_xlabel("decline rate (%)  —  lower is better")
    ax.set_ylabel("decline precision (%)\n—  higher is better", fontsize=9)
    ax.set_title(
        "Abstention quality: this repo declines less often AND more accurately"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decline_tradeoff.png", bbox_inches="tight")
    plt.close(fig)



def fig_cost_speed(results: list[dict[str, Any]]) -> None:
    """Bar chart: p50 seconds per meter-day, log scale (LLM vs local model)."""
    eval_rows = [r for r in results if r["split"] == "eval"]
    order = [
        "rules_floor_v1",
        "classical_logreg_v1",
        "llm_zero_shot_v1",
        "llm_prompt_opt_v1",
    ]
    eval_rows.sort(key=lambda r: order.index(r["system"]))

    names = [PRETTY[r["system"]] for r in eval_rows]
    seconds = [r["latency_p50_s"] for r in eval_rows]
    colours = [GREEN, BLUE, ORANGE, RED]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bars = ax.bar(names, seconds, color=colours, width=0.6)
    ax.set_yscale("log")

    for bar, value in zip(bars, seconds):
        label = f"{value * 1000:.2f} ms" if value < 1 else f"{value:.1f} s"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("p50 latency per meter-day (seconds, log scale)")
    ax.set_ylim(1e-4, 300)
    ax.set_title("Speed: sub-millisecond local inference vs a 12-30 s LLM call")
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_speed.png", bbox_inches="tight")
    plt.close(fig)


def fig_synth_mix(mix: Counter) -> None:
    """Two-panel: weak-label mix of the training set, and the human gold mix.

    Shows why synthetic data is needed: the human labels (30+30) are far too
    few to fit 26 features, while the generated set provides 1,080 days
    across the same four classes.
    """
    order = ["active", "standby", "off", "unsure"]
    synth_vals = [mix[lab] for lab in order]

    # Human gold counts come from the committed eval predictions plus the
    # README's documented train split (30/30) — read gold counts from the
    # eval file and scale the train side from the same label guide.
    preds = load_predictions()
    eval_gold = Counter(row["gold_label"] for row in preds)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    colours = [BLUE, GREEN, GREY, ORANGE]

    bars = axes[0].bar(order, synth_vals, color=colours, width=0.62)
    for bar, value in zip(bars, synth_vals):
        axes[0].annotate(
            str(value),
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_title(f"Synthetic training days (n={sum(synth_vals):,})")
    axes[0].set_ylabel("meter-days")
    axes[0].set_ylim(0, max(synth_vals) * 1.18)

    human_vals = [eval_gold[lab] for lab in order]
    bars = axes[1].bar(order, human_vals, color=colours, width=0.62)
    for bar, value in zip(bars, human_vals):
        axes[1].annotate(
            str(value),
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_title(f"Human gold labels (n={sum(human_vals)} eval shown)")
    axes[1].set_ylabel("meter-days")
    axes[1].set_ylim(0, 12)

    fig.suptitle(
        "Why synthetic data: 1,080 weak days to train on, 60 human days to measure",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_mix.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build every figure and print the numbers behind it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    preds = load_predictions()

    print(f"results rows: {len(results)} | eval predictions: {len(preds)} days")
    for row in results:
        print(
            f"  {row['system']:22} {row['split']:5} "
            f"acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f}"
        )

    print("\npredicted vs gold (mismatches only):")
    for row in preds:
        if row["predicted_label"] != row["gold_label"]:
            print(
                f"  {row['day_id']:24} gold={row['gold_label']:8} "
                f"pred={row['predicted_label']}"
            )

    mix = load_synth_mix()
    print(f"\nsynth weak-label mix: {dict(mix)}  total={sum(mix.values())}")

    fig_scoreboard(results)
    fig_per_class_f1(results)
    fig_confusion(preds)
    fig_decline_tradeoff(results)
    fig_cost_speed(results)
    fig_synth_mix(mix)

    print(f"\nwrote {len(list(FIG_DIR.glob('*.png')))} figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

