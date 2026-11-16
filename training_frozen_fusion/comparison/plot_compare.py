"""Charts for the joint comparison (reads results/comparison.json).

    python training/comparison/plot_compare.py
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
from plot_results import GRID, INK, INK_2, S1, S2, S3, S4, SURFACE, heb  # noqa: E402  (experiment_1 style)

PLOTS = HERE / "results" / "plots"
MODELS = ("exp1", "exp1-slots", "exp2", "exp3")
COLOR = {"exp3": S1, "exp2": S2, "exp1-slots": S3, "exp1": S4}  # fixed per entity
LABEL = {"exp1": "Exp1 · E5 · 77 fields", "exp1-slots": "Exp1-slots · E5", "exp2": "Exp2 · TF-IDF",
         "exp3": "Exp3 · E5 ⊕ TF-IDF"}
D = json.loads((HERE / "results" / "comparison.json").read_text(encoding="utf-8"))
SEL = D["selected"]
CONFIG = (f"selected on val_eval: Exp1 C=64 · Exp1-slots C={SEL['exp1-slots']['C']} · Exp2 C={SEL['exp2']['C']} · "
          f"Exp3 C={SEL['exp3']['C']}, w={SEL['exp3']['w']}")


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", PLOTS / name)


def footer(fig):
    fig.text(0.01, 0.0, CONFIG, fontsize=8, color=INK_2, ha="left", va="top", transform=fig.transFigure)
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.16))


def plot_headline():
    metrics = [("slot_exact_set", "Slot exact-set (primary)"), ("strict_deployable", "Strict field exact-set"),
               ("slot_micro_f1", "Slot micro-F1"), ("domain_accuracy", "Domain accuracy")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    y = np.arange(len(MODELS))
    for ax, (key, title) in zip(axes, metrics):
        for split, marker, fill in (("val_eval", "o", False), ("test", "o", True)):
            for i, m in enumerate(MODELS):
                h = D["headline"][split][m]
                v = h["strict_exact_set"]["deployable"] if key == "strict_deployable" else h[key]
                ax.plot(v, i, marker=marker, markersize=8, color=COLOR[m], markerfacecolor=COLOR[m] if fill else SURFACE,
                        markeredgewidth=2, linestyle="none")
                if split == "test":
                    ax.annotate(f"{v:.3f}", (v, i), xytext=(0, 9), textcoords="offset points", ha="center",
                                fontsize=8, color=INK)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.15)
    axes[0].set_yticks(y, [LABEL[m] for m in MODELS])
    axes[0].set_ylim(len(MODELS) - 0.5, -0.8)  # inverted, with headroom for value labels
    axes[-1].plot([], [], "o", color=INK_2, markerfacecolor=SURFACE, markeredgewidth=2, label="val_eval (development)")
    axes[-1].plot([], [], "o", color=INK_2, label="test (frozen configs)")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Headline metrics: hollow = val_eval (development), filled = test", x=0.01, ha="left", color=INK,
                 fontsize=12)
    fig.tight_layout()
    footer(fig)
    save(fig, "headline.png")


def plot_bootstrap():
    pairs = list(D["bootstrap"]["test"].keys())
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, (split, dy, fill) in enumerate((("val_eval", -0.15, False), ("test", 0.15, True))):
        for i, p in enumerate(pairs):
            b = D["bootstrap"][split][p]
            lo, hi = b["ci95"]
            ax.plot([lo, hi], [i + dy, i + dy], color=S1, linewidth=2)
            ax.plot(b["diff"], i + dy, "o", color=S1, markerfacecolor=S1 if fill else SURFACE, markeredgewidth=2,
                    markersize=7, label=("test" if fill else "val_eval (development)") if i == 0 else None)
            ax.annotate(f"{b['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]", (hi, i + dy), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(pairs)), [f"{p}\n({D['bootstrap']['test'][p]['question']})" for p in pairs], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.02, 0.27)
    ax.set_xlabel("difference in slot exact-set accuracy (A − B)")
    ax.set_title("Paired differences, family-level bootstrap (2,000 replicates, 95% CI)", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    footer(fig)
    save(fig, "bootstrap_ci.png")


def plot_curves():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m in ("exp1-slots", "exp2", "exp3"):
        pts = D["learning_curves"][m]["points"]
        n = np.array([np.mean([r["size"] for r in p["runs"]]) for p in pts])
        mean = np.array([np.mean([r["slot_exact_set"] for r in p["runs"]]) for p in pts])
        std = np.array([np.std([r["slot_exact_set"] for r in p["runs"]]) for p in pts])
        ax.fill_between(n, mean - std, mean + std, color=COLOR[m], alpha=0.18, linewidth=0)
        ax.plot(n, mean, marker="o", color=COLOR[m], markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABEL[m])
        ax.annotate(f"{mean[-1]:.3f}", (n[-1], mean[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([250, 500, 1000, 2000, 4000, 6171], ["250", "500", "1k", "2k", "4k", "6.2k"])
    ax.minorticks_off()
    ax.set_xlabel("training rows (log)")
    ax.set_ylabel("slot exact-set (val_eval, development)")
    ax.set_title("Learning curves at each experiment's selected config (mean ± std, 3 seeds)", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    footer(fig)
    save(fig, "learning_curves.png")


def plot_slices(split="test"):
    keys = [("language", "Language register"), ("operator", "Operator"), ("length", "Query length"),
            ("rephrasing", "Rephrasing family"), ("quality", "Quality tier")]
    fig, axes = plt.subplots(1, len(keys), figsize=(16, 3.8), sharey=True)
    for ax, (key, title) in zip(axes, keys):
        groups = list(D["slices"][split]["exp1"][key].keys())
        x = np.arange(len(groups))
        wbar = 0.19
        for k, m in enumerate(MODELS):
            v = [D["slices"][split][m][key][g]["slot_exact_set"] for g in groups]
            ax.bar(x + (k - 1.5) * (wbar + 0.01), v, width=wbar, color=COLOR[m], label=LABEL[m])
        ns = [D["slices"][split]["exp1"][key][g]["n"] for g in groups]
        ax.set_xticks(x, [f"{g}\nn={n}" for g, n in zip(groups, ns)], fontsize=8)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel(f"slot exact-set ({split})")
    handles, labels = axes[0].get_legend_handles_labels()
    title = f"Slot exact-set by slice ({split}; " + ("development" if split == "val_eval" else "frozen configs") + ")"
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=4, fontsize=9)
    footer(fig)
    save(fig, f"slices_{split}.png")


def plot_risk_coverage(split="test"):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in MODELS:
        rc = D["confidence"][split][m]["prediction_margin"]["risk_coverage"]
        ax.plot([r["coverage"] for r in rc], [r["slot_exact_set"] for r in rc], marker="o", color=COLOR[m],
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"{LABEL[m]} · margin (AUROC {D['confidence'][split][m]['prediction_margin']['auroc']:.2f})")
    rc = D["confidence"][split]["exp3"]["domain_confidence"]["risk_coverage"]
    ax.plot([r["coverage"] for r in rc], [r["slot_exact_set"] for r in rc], color=COLOR["exp3"],
            linestyle=(0, (3, 3)), linewidth=1.5,
            label=f"Exp3 · domain confidence (AUROC {D['confidence'][split]['exp3']['domain_confidence']['auroc']:.2f})")
    ax.set_xlabel("coverage (share of queries kept, most confident first)")
    ax.set_ylabel(f"slot exact-set on kept queries ({split})")
    ax.set_title("Which score predicts a fully correct slot set?", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.invert_xaxis()
    footer(fig)
    save(fig, f"risk_coverage_{split}.png")


def plot_latency():
    lat = D["latency"]["models"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(MODELS))
    left = np.zeros(len(MODELS))
    for stage, color in (("featurize", S1), ("classify", S2), ("postprocess", S3)):
        v = np.array([lat[m]["batch1_ms_p50_by_stage"][stage] for m in MODELS])
        ax.barh(y, v, left=left, color=color, height=0.6, label=stage, edgecolor=SURFACE, linewidth=2)
        left += v
    for i, m in enumerate(MODELS):
        ax.annotate(f"p50 {lat[m]['batch1_ms_p50']:.1f} ms · p95 {lat[m]['batch1_ms_p95']:.1f} · "
                    f"batch-32 {lat[m]['batch32_ms_per_query']:.2f} ms/q", (left[i], i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [LABEL[m] for m in MODELS])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, left.max() * 2.1)
    ax.set_xlabel("end-to-end latency per query, batch 1, median (ms, CPU)")
    ax.set_title(f"Inference latency by stage (E5 load {D['latency']['e5_model_load_seconds']} s, excluded)",
                 loc="left")
    ax.legend(loc="lower right", fontsize=8)
    footer(fig)
    save(fig, "latency.png")


def plot_slot_delta(pair="exp3 vs exp1-slots", k=12):
    d = D["per_slot_delta"][pair]
    rows = d["improved"][:k] + d["regressed"][:6]
    rows = sorted(rows, key=lambda r: r["delta"])
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(rows) + 1.4))
    y = np.arange(len(rows))
    ax.barh(y, [r["delta"] for r in rows], color=[S1 if r["delta"] > 0 else S2 for r in rows], height=0.7)
    ax.set_yticks(y, [heb(r["slot"]) for r in rows], fontsize=8.5)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['delta']:+.2f} (n={r['support']})", (r["delta"], i), xytext=(-4 if r["delta"] > 0 else 4, 0),
                    textcoords="offset points", va="center", ha="right" if r["delta"] > 0 else "left", fontsize=8,
                    color=INK_2)
    # RTL layout: Hebrew slot names on the right
    ax.yaxis.tick_right()
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    lim = max(abs(r["delta"]) for r in rows) * 1.35
    ax.set_xlim(lim, -lim)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 change (val_eval, development) ← better · worse →")
    ax.set_title(f"Per-slot F1: {pair} ({d['n_improved']} improved, {d['n_regressed']} regressed)", loc="right")
    footer(fig)
    save(fig, "slot_delta.png")


def main():
    plt.rcParams["axes.edgecolor"] = GRID
    plot_headline()
    plot_bootstrap()
    plot_curves()
    plot_slices("test")
    plot_slices("val_eval")
    plot_risk_coverage("test")
    plot_latency()
    plot_slot_delta()


if __name__ == "__main__":
    main()
