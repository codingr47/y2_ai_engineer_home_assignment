"""Render the experiment's result JSONs as PNG charts in artifacts/results/plots/.

    python plot_results.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import RESULTS_DIR  # noqa: E402

PLOTS = RESULTS_DIR / "plots"

# reference palette (dataviz skill, light mode): categorical slots 1-4 in fixed order + surfaces/ink
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

METRICS = [("micro_f1", "Micro-F1"), ("macro_f1", "Macro-F1"),
           ("exact_set_accuracy", "Exact field-set acc."), ("domain_accuracy", "Domain accuracy")]

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False, "legend.labelcolor": INK,
})


def heb(s: str) -> str:
    """matplotlib has no bidi: reverse pure-Hebrew labels so they read right-to-left."""
    return s[::-1]


def load(name):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def curve_stats(curve, key, variant="full"):
    n, mean, std = [], [], []
    for point in curve:
        vals = [r["variants"][variant][key] for r in point["runs"]]
        n.append(np.mean([r["size"] for r in point["runs"]]))
        mean.append(np.mean(vals))
        std.append(np.std(vals))
    return np.array(n), np.array(mean), np.array(std)


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOTS / name}")


def plot_learning_curve():
    series = [("learning_curve_C64.json", "C = 64", S1), ("learning_curve_C1.json", "C = 1 (default)", S2)]
    series = [(load(f), label, c) for f, label, c in series if (RESULTS_DIR / f).exists()]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
    for ax, (key, title) in zip(axes, METRICS):
        for curve, label, color in series:
            n, m, s = curve_stats(curve, key)
            ax.fill_between(n, m - s, m + s, color=color, alpha=0.18, linewidth=0)
            ax.plot(n, m, color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
            ax.annotate(f"{m[-1]:.2f}", (n[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set_xticks([250, 500, 1000, 2000, 4000, 6171], ["250", "500", "1k", "2k", "4k", "6.2k"])
        ax.minorticks_off()
        ax.set_title(title, loc="left")
        ax.set_xlabel("training rows (log)")
    axes[0].set_ylabel("val_eval score")
    axes[0].legend(loc="lower right")
    fig.suptitle("Learning curve: field and domain metrics vs. training-set size (mean ± std, 3 seeds)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "learning_curve.png")


def plot_c_sweep():
    sweep = load("c_sweep.json")
    Cs = sorted(sweep, key=float)
    x = np.array([float(c) for c in Cs])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (key, title), color in zip(METRICS, (S1, S2, S3, S4)):
        y = np.array([sweep[c][0]["variants"]["full"][key] for c in Cs])
        ax.plot(x, y, color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5, label=title)
        ax.annotate(title, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.axvline(64, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.annotate("chosen C = 64", (64, ax.get_ylim()[0]), xytext=(4, 6), textcoords="offset points",
                fontsize=9, color=INK_2)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{c:g}" for c in x])
    ax.minorticks_off()
    ax.set_xlabel("C (inverse L2 regularisation, log)")
    ax.set_ylabel("val_eval score (full train)")
    ax.set_title("Regularisation sweep: C = 1 over-regularises unit-norm embeddings", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(x[0] / 1.5, x[-1] * 6)
    save(fig, "c_sweep.png")


def plot_calibration(min_n=10):
    res = load("final_test.json")
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(6, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (3, 3)), label="perfect calibration")
    variants = (("raw", "raw LR", S2, -0.006), ("full", "temperature-scaled", S1, 0.006))
    for variant, label, color, dx in variants:
        bins = [b for b in res["variants"][variant]["domain"]["reliability"]]
        kept = [b for b in bins if b["n"] >= min_n]
        ax.plot([b["conf"] for b in kept], [b["acc"] for b in kept], color=color, marker="o",
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"{label} (ECE {res['variants'][variant]['domain_ece']:.3f})")
        lo = np.array([float(b["bin"].split("-")[0]) for b in bins])
        axn.bar(lo + 1 / 30 + dx, [b["n"] for b in bins], width=0.011, color=color)
    ax.set_xlim(0.3, 1.01)
    ax.set_ylim(0.2, 1.02)
    ax.set_ylabel("observed domain accuracy")
    ax.set_title(f"Domain confidence reliability on test (bins with n ≥ {min_n})", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    axn.set_yscale("log")
    axn.set_ylabel("queries per bin")
    axn.set_xlabel("predicted domain confidence")
    axn.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "domain_calibration.png")


def plot_routing():
    res = load("final_test.json")
    routing = res["variants"]["full"]["domain"]["routing"]
    cuts = [r["min_conf"] for r in routing]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for ax, key, title, color in ((axes[0], "coverage", "Share of queries auto-routed", S1),
                                  (axes[1], "accuracy", "Domain accuracy on routed queries", S1)):
        y = [r[key] for r in routing]
        ax.plot(cuts, y, color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5)
        for c, v in zip(cuts, y):
            ax.annotate(f"{v:.1%}", (c, v), xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK)
        ax.set_title(title, loc="left")
        ax.set_xlabel("minimum calibrated confidence")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    axes[0].set_ylim(0.8, 1.02)
    axes[1].set_ylim(0.94, 1.0)
    fig.suptitle("Routing trade-off on test (temperature-scaled domain head)", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "routing_tradeoff.png")


def plot_field_recall(k=20):
    res = load("final_test.json")
    rows = res["variants"]["full"]["per_field"][:k][::-1]  # worst at the top
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * k + 1.2))
    y = np.arange(len(rows))
    ax.barh(y, [r["recall"] for r in rows], color=S1, height=0.7)
    ax.set_yticks(y, [heb(r["field"]) for r in rows], fontsize=9)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['recall']:.2f}  (n={r['support']})", (r["recall"], i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("recall on test")
    ax.set_title(f"{k} fields with the lowest recall (test, C = 64)", loc="left")
    save(fig, "field_recall.png")


def main():
    plot_learning_curve()
    plot_c_sweep()
    plot_calibration()
    plot_routing()
    plot_field_recall()


if __name__ == "__main__":
    main()
