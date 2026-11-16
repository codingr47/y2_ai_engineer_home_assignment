"""Learning curve over training-set size (+ optional C sweep and final test run).

    python training/experiment_1/run_experiment.py                 # learning curve on val_eval
    python training/experiment_1/run_experiment.py --c-grid 1 4 16 64
    python training/experiment_1/run_experiment.py --final-test --C 16
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import joblib
import numpy as np

from config import CURVE_SEEDS, MODELS_DIR, RESULTS_DIR, TRAIN_SIZES
from evaluate import HEADLINE, evaluate
from train import VARIANTS, Heads, load_split, load_vocab


def family_subsample(train, size, seed):
    """~size rows, whole families only, domain proportions kept."""
    if size is None or size >= len(train.labels):
        return np.arange(len(train.labels))
    rng = np.random.default_rng(seed)
    fam_rows = defaultdict(list)
    for i, m in enumerate(train.meta):
        fam_rows[m["family"]].append(i)
    by_dom = defaultdict(list)
    for fam, rows in fam_rows.items():
        by_dom[train.labels[rows[0]]["domain_label"]].append(fam)
    idx = []
    for dom, fams in sorted(by_dom.items()):
        quota = round(size * sum(len(fam_rows[f]) for f in fams) / len(train.labels))
        taken = []
        for f in rng.permutation(fams):
            if len(taken) >= quota:
                break
            taken += fam_rows[f]
        idx += taken
    return np.array(sorted(idx))


def run_one(train, calib, ev, C, size, seed, variants=VARIANTS, detailed=False):
    sub = train.subset(family_subsample(train, size, seed))
    t0 = time.time()
    heads = Heads(vocab=load_vocab(), C=C).fit(sub).calibrate(calib)
    out = {"size": len(sub.labels), "target_size": size or "all", "seed": seed, "C": C,
           "fit_seconds": round(time.time() - t0, 1), "constant_fields": len(heads.constant_fields),
           "domain_cal_method": heads.domain_cal.method, "field_global_thr": heads.field_cal.global_thr,
           "variants": {v: evaluate(heads, ev, v, detailed=detailed) for v in variants}}
    return out, heads


def agg(runs, variant):
    rows = [r["variants"][variant] for r in runs]
    return {k: (float(np.mean([x[k] for x in rows])), float(np.std([x[k] for x in rows]))) for k in HEADLINE}


def fmt_table(groups, variant, key_name):
    head = f"| {key_name} | n_train | " + " | ".join(HEADLINE) + " |"
    lines = [head, "|" + "---|" * (len(HEADLINE) + 2)]
    for key, runs in groups:
        a = agg(runs, variant)
        n = int(np.mean([r["size"] for r in runs]))
        cells = [f"{m:.3f}" + (f" ±{s:.3f}" if len(runs) > 1 else "") for m, s in (a[k] for k in HEADLINE)]
        lines.append(f"| {key} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=int, default=None, help="train sizes (omit for defaults; full added)")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--c-grid", nargs="*", type=float, default=None, help="C sweep on full train")
    ap.add_argument("--final-test", action="store_true", help="fit on full train, report on test, save models")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train, calib, ev = load_split("train"), load_split("val_calib"), load_split("val_eval")

    if args.final_test:
        test = load_split("test")
        res, heads = run_one(train, calib, test, args.C, None, 0, detailed=True)
        gold = [i for i, m in enumerate(test.meta) if m["quality"] == "gold"]
        g = test.subset(gold)
        g.name = "test_gold"
        res["gold"] = {v: evaluate(heads, g, v, detailed=False) for v in VARIANTS}
        res["val_eval"] = {v: evaluate(heads, ev, v, detailed=False) for v in VARIANTS}
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(heads, MODELS_DIR / "heads.joblib")
        (MODELS_DIR / "thresholds.json").write_text(json.dumps(
            {"global": heads.field_cal.global_thr, "per_field": dict(zip(heads.vocab, heads.field_cal.thr.tolist())),
             "domain_calibration": heads.domain_cal.method, "domain_cv_logloss": heads.domain_cal.cv_logloss},
            ensure_ascii=False, indent=1), encoding="utf-8")
        (RESULTS_DIR / "final_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        for name, block in (("test", res["variants"]), ("test_gold", res["gold"]), ("val_eval", res["val_eval"])):
            print(f"\n## {name} (C={args.C})")
            print(fmt_table([(v, [{"size": res["size"], "variants": {"_": block[v]}}]) for v in VARIANTS], "_", "variant"))
        return

    if args.c_grid:
        runs = {}
        for C in args.c_grid:
            r, _ = run_one(train, calib, ev, C, None, 0, detailed=False)
            runs[C] = [r]
            print(f"C={C}: " + ", ".join(f"{k}={r['variants']['full'][k]:.3f}" for k in HEADLINE), flush=True)
        (RESULTS_DIR / "c_sweep.json").write_text(json.dumps({str(k): v for k, v in runs.items()},
                                                             ensure_ascii=False, indent=1), encoding="utf-8")
        md = "\n\n".join(f"### variant `{v}`\n" + fmt_table([(f"C={c}", rs) for c, rs in runs.items()], v, "C")
                         for v in VARIANTS)
        (RESULTS_DIR / "c_sweep.md").write_text(md, encoding="utf-8")
        print("\n" + md)
        return

    sizes = list(args.sizes) + [None] if args.sizes else list(TRAIN_SIZES)
    curve = []
    for size in sizes:
        seeds = CURVE_SEEDS if size is not None else (0,)
        runs = []
        for s in seeds:
            r, _ = run_one(train, calib, ev, args.C, size, s, detailed=(size is None))
            runs.append(r)
            f = r["variants"]["full"]
            print(f"size={size or 'all'} seed={s} n={r['size']} dom={f['domain_accuracy']:.3f} "
                  f"microF1={f['micro_f1']:.3f} macroF1={f['macro_f1']:.3f} exact={f['exact_set_accuracy']:.3f} "
                  f"({r['fit_seconds']}s)", flush=True)
        curve.append((size or "all", runs))
    tag = f"C{args.C:g}"
    (RESULTS_DIR / f"learning_curve_{tag}.json").write_text(
        json.dumps([{"size": k, "runs": v} for k, v in curve], ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# Learning curve (C={args.C:g}, eval = val_eval, mean ± std over seeds {CURVE_SEEDS})"]
    md += [f"\n### variant `{v}`\n" + fmt_table(curve, v, "size") for v in VARIANTS]
    full = curve[-1][1][0]["variants"]["full"]
    md.append("\n### Lowest per-field recall (full train, variant `full`)\n\n| field | recall | precision | support |\n|---|---|---|---|")
    md += [f"| {d['field']} | {d['recall']:.2f} | {d['precision']:.2f} | {d['support']} |" for d in full["per_field"][:15]]
    (RESULTS_DIR / f"learning_curve_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
