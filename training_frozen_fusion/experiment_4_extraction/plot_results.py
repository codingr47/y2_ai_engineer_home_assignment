"""Charts for the extraction evaluation (reads results/summary_val_eval.json).

    python training_frozen_fusion/experiment_4_extraction/plot_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import paths  # noqa: E402,F401

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_results import GRID, INK, INK_2, S1, S2, S3, S4, SURFACE  # noqa: E402  (experiment_1 style)

PLOTS = HERE / "results" / "plots"
D = json.loads((HERE / "results" / "summary_val_eval.json").read_text(encoding="utf-8"))
MAIN = "hybrid"
FOOT = "val_eval (development) · labels v2 · classifier: frozen Exp3 v2 · extractor: " + MAIN


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, -0.04, FOOT, fontsize=8, color=INK_2, ha="left", va="top")
    fig.savefig(PLOTS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", PLOTS / name)


def plot_waterfall():
    w = D["variants"][MAIN]["scenario3"]["waterfall"]
    n = w["rows"]
    rescued = w["rescued_rows (field set wrong, JSON exact: spurious field had no value)"]
    stages = [("all rows", n, S4), ("domain correct", w["domain_correct"], S3),
              ("field set exact", w["field_set_exact"], S2),
              ("…and all values right", w["json_exact_on_field_set_exact_rows"], S1),
              ("JSON exact (incl. rescued)", w["json_exact"], S1)]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(stages))
    for i, (label, v, color) in enumerate(stages):
        if i == len(stages) - 1:
            ax.barh(i, w["json_exact_on_field_set_exact_rows"], color=color, height=0.6)
            ax.barh(i, rescued, left=w["json_exact_on_field_set_exact_rows"], color=color, alpha=0.45, height=0.6,
                    hatch="//", edgecolor=SURFACE)
        else:
            ax.barh(i, v, color=color, height=0.6)
        ax.annotate(f"{v} ({v / n:.1%})", (v, i), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.annotate(f"+{rescued} rescued: a spurious predicted field had no value, so it was dropped",
                (w["json_exact"], len(stages) - 1), xytext=(6, -14), textcoords="offset points", fontsize=8,
                color=INK_2)
    ax.set_yticks(y, [s[0] for s in stages])
    ax.invert_yaxis()
    ax.set_xlim(0, n * 1.32)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("queries")
    ax.set_title("Where complete-JSON accuracy is won and lost", loc="left")
    save(fig, "extraction_waterfall.png")


def plot_gap():
    gap = D["variants"][MAIN]["scenario3"]["gap_field_set_to_json"]
    cats = [(c, v) for c, v in gap["by_primary_category"].items() if v]
    cats.sort(key=lambda kv: -kv[1])
    group_color = {"span detection": S1, "numeric interpretation": S2, "enum normalization": S3,
                   "other extraction": S4}
    from evaluate_extraction import CATEGORY_GROUP
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(cats) + 1.4))
    y = np.arange(len(cats))
    ax.barh(y, [v for _, v in cats], color=[group_color[CATEGORY_GROUP[c]] for c, _ in cats], height=0.65)
    for i, (c, v) in enumerate(cats):
        ax.annotate(f"{v}  ·  {CATEGORY_GROUP[c]}", (v, i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [c.replace("_", " ") for c, _ in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, max(v for _, v in cats) * 1.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("rows (first failing value category)")
    ax.set_title(f"Field set right, JSON wrong: {gap['rows']} rows = {gap['points']} points", loc="left")
    save(fig, "extraction_gap.png")


def plot_scenario1():
    variants = list(D["variants"])
    types = ["enum", "numeric", "bool", "date", "text"]
    colors = {"lexicon": S3, "hybrid": S1, "hybrid+morph": S2}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(types))
    wbar = 0.8 / len(variants)
    for k, v in enumerate(variants):
        bt = D["variants"][v]["scenario1"]["by_type"]
        vals = [bt[t]["accuracy"] for t in types]
        ax.bar(x + (k - (len(variants) - 1) / 2) * wbar, vals, width=wbar - 0.02, color=colors.get(v, S4),
               label=f"{v} (all: {D['variants'][v]['scenario1']['value_accuracy']:.3f})")
    n = D["variants"][variants[0]]["scenario1"]["by_type"]
    ax.set_xticks(x, [f"{t}\nn={n[t]['n']}" for t in types])
    ax.set_ylim(0.75, 1.01)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("value accuracy")
    ax.set_title("Scenario 1 — gold fields supplied: value accuracy by field type", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    save(fig, "extraction_scenario1.png")


if __name__ == "__main__":
    plt.rcParams["axes.edgecolor"] = GRID
    sys.path.insert(0, str(HERE))
    plot_waterfall()
    plot_gap()
    plot_scenario1()
