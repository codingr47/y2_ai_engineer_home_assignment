"""Error analysis of Experiment 1's field classifier (read-only on the model and the data).

Uses the saved C=64 heads (artifacts/models/heads.joblib) exactly as trained and calibrated: no refit, no new
thresholds. Predictions are made on val_eval, the model-validation half of val. The test split is never loaded;
train is read only for support counts, span statistics and the kNN embedding probe.

    python training/training_frozen_fusion/experiment_1/experiment_analysis/analyze.py

Outputs (this folder):
    predictions_val_eval.jsonl   one record per query (ids, query, gold/pred domain + fields, all 77 probabilities,
                                 FP/FN fields, metadata, generation batch)
    predictions_val_eval.csv     same, flat (one p__<field> column per field)
    diagnostics.json / .md       aggregate diagnostics
    root_cause_sample.jsonl      ~100 stratified cases for manual root-cause review
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import joblib  # noqa: E402

from config import DATA_DIR, MODELS_DIR  # noqa: E402
from train import Heads, domain_mask, load_split  # noqa: E402,F401  (Heads needed to unpickle)

EXPECTED_C = 64.0
VARIANT = "full"  # the deployed inference path: calibrated probs, tuned thresholds, predicted-domain mask
GENERATOR_FILE = EXP.parents[3] / "y2_ai_synthetic_generator" / "data" / "synthetic.jsonl"
KNN_K = 10
SAMPLE_SEED = 0


# ------------------------------------------------------------------ helpers
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def short(field: str) -> str:
    return field.split("/", 1)[1]


def generation_batches() -> dict[str, dict]:
    """id -> {batch, generator} parsed from the generator's notes tag, e.g.
    '[synthetic ambiguous/typo gap_fill gen:z-ai/glm-5.3]'. Empty dict if the file is unavailable."""
    if not GENERATOR_FILE.exists():
        return {}
    out = {}
    for r in read_jsonl(GENERATOR_FILE):
        m = re.match(r"\[([^\]]*)\]", r.get("notes") or "")
        toks = m.group(1).split()[2:] if m else []
        gen = next((t[4:] for t in toks if t.startswith("gen:")), None)
        source = next((t.split(":", 1)[1] for t in toks if t.startswith("rephrase_of:")), None)
        # a rephrasing is tagged 'rephrase_of:<src> variation:<kind>'; report the variation kind as its batch
        kinds = [t for t in toks if not t.startswith(("gen:", "rephrase_of:"))]
        kind = next((t for t in kinds if t.startswith("variation:")), kinds[0] if kinds else "seed")
        out[r["id"]] = {"batch": kind, "batch_family": kind.split(":")[0], "generator": gen,
                        "rephrase_of": source}
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def acc_table(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key] if not isinstance(r[key], list) else "+".join(r[key])].append(r)
    out = {}
    for k, rs in sorted(g.items(), key=lambda kv: str(kv[0])):
        out[str(k)] = {
            "n": len(rs),
            "exact_set_accuracy": round(float(np.mean([x["exact_set"] for x in rs])), 4),
            "domain_accuracy": round(float(np.mean([x["domain_correct"] for x in rs])), 4),
            "fn_per_query": round(float(np.mean([len(x["false_negatives"]) for x in rs])), 3),
            "fp_per_query": round(float(np.mean([len(x["false_positives"]) for x in rs])), 3),
        }
    return out


# ------------------------------------------------------------------ main
def main():
    model_path, thr_path = MODELS_DIR / "heads.joblib", MODELS_DIR / "thresholds.json"
    hashes_before = {"heads.joblib": sha(model_path), "thresholds.json": sha(thr_path)}

    heads = joblib.load(model_path)
    assert heads.C == EXPECTED_C, f"saved model has C={heads.C}, analysis expects C={EXPECTED_C}"
    vocab = heads.vocab
    thr = heads.field_cal.thr
    platt = heads.field_cal.platt

    ev = load_split("val_eval")
    train = load_split("train")
    raw = {r["id"]: r for r in read_jsonl(DATA_DIR / "val.jsonl")}
    raw_train = read_jsonl(DATA_DIR / "train.jsonl")
    batches = generation_batches()

    # ---- predictions (deployed path) + unmasked calibrated probabilities
    Pd, Pf_masked, pred = heads.predict(ev.X, VARIANT)
    Pf = heads.field_cal.transform(ev.X)  # calibrated, before the domain mask
    classes = heads.domain_model.classes_
    pred_dom = classes[Pd.argmax(1)]
    Y = heads.mlb.transform(ev.fields).astype(bool)
    fidx = {f: j for j, f in enumerate(vocab)}

    # ---- kNN probe: does the embedding neighbourhood carry the field? (train neighbours, cosine)
    sims = ev.X @ train.X.T
    nn = np.argsort(-sims, axis=1)[:, :KNN_K]
    Ytr = heads.mlb.transform(train.fields).astype(bool)
    knn_rate = Ytr[nn].mean(1)  # (n_eval, 77): share of the 10 nearest train queries that carry each field

    records = []
    for i, (lab, meta) in enumerate(zip(ev.labels, ev.meta)):
        r = raw[lab["id"]]
        gold = lab["field_labels"]
        predicted = [vocab[j] for j in np.flatnonzero(pred[i])]
        fn = [f for f in gold if f not in predicted]
        fp = [f for f in predicted if f not in gold]
        md = r.get("metadata") or {}
        b = batches.get(lab["id"], {})
        spans = r.get("spans") or {}
        records.append({
            "id": lab["id"],
            "query": meta["query"],
            "gold_domain": lab["domain_label"],
            "pred_domain": str(pred_dom[i]),
            "domain_confidence": round(float(Pd[i].max()), 4),
            "domain_correct": lab["domain_label"] == pred_dom[i],
            "gold_fields": gold,
            "pred_fields": predicted,
            "false_negatives": fn,
            "false_positives": fp,
            "exact_set": not fn and not fp,
            "n_expected_fields": len(gold),
            "gold_params": (r.get("expected") or {}).get("params") or {},
            "gold_spans": {k: [{"text": s["text"], "op": s["op"], "origin": s.get("origin")} for s in v]
                           for k, v in spans.items()},
            "field_probabilities": {f: round(float(Pf[i, j]), 4) for j, f in enumerate(vocab)},
            "language": (md.get("language") or [None])[0],
            "complexity": (md.get("complexity") or [None])[0],
            "operator": md.get("operator") or [],
            "field_count_tag": (md.get("field_count") or [None])[0],
            "quality": r.get("quality"),
            "generation_batch": b.get("batch"),
            "generation_batch_family": b.get("batch_family"),
            "generator": b.get("generator"),
            "rephrase_of": b.get("rephrase_of"),
            "family": r.get("family"),
            "is_rephrasing": r.get("family") not in (None, lab["id"]),
            "_i": i,
        })

    write_predictions(records, vocab)
    whatif = what_if(records, Pf, thr, fidx)

    # ---- per-field table
    fn_c, fp_c, tp_c = Counter(), Counter(), Counter()
    fn_domain_err, fp_domain_err = Counter(), Counter()
    for rec in records:
        for f in rec["gold_fields"]:
            if f in rec["false_negatives"]:
                fn_c[f] += 1
                if not rec["domain_correct"]:
                    fn_domain_err[f] += 1
            else:
                tp_c[f] += 1
        for f in rec["false_positives"]:
            fp_c[f] += 1
            if not rec["domain_correct"]:
                fp_domain_err[f] += 1
    train_support = Ytr.sum(0)
    span_ops = span_op_stats(raw_train)
    fields = []
    for j, f in enumerate(vocab):
        p, r_, f1 = prf(tp_c[f], fp_c[f], fn_c[f])
        fields.append({
            "field": f, "support": int(Y[:, j].sum()), "train_support": int(train_support[j]),
            "tp": tp_c[f], "fn": fn_c[f], "fp": fp_c[f], "errors": fn_c[f] + fp_c[f],
            "fn_from_domain_error": fn_domain_err[f], "fp_from_domain_error": fp_domain_err[f],
            "precision": round(p, 4), "recall": round(r_, 4), "f1": round(f1, 4),
            "threshold": round(float(thr[j]), 3), "platt_calibrated": bool(platt[j]),
            "train_span_ops": span_ops.get(f, {}),
        })
    fields.sort(key=lambda d: (-d["errors"], -d["fn"]))

    # ---- confusions (rows with the correct domain only: field-level, not domain-level, confusion)
    pair_c, name_pair_c = Counter(), Counter()
    for rec in records:
        if not rec["domain_correct"]:
            continue
        for a in rec["false_negatives"]:
            for b_ in rec["false_positives"]:
                pair_c[(a, b_)] += 1
                name_pair_c[(short(a), short(b_))] += 1
    # co-occurrence of a field with its range twin in train (same query labelled with both)
    confusions = [{"missed": a, "predicted_instead": b_, "count": c} for (a, b_), c in pair_c.most_common(25)]

    # ---- FN scores vs thresholds
    fn_scores = []
    for i, rec in enumerate(records):
        for f in rec["false_negatives"]:
            j = fidx[f]
            fn_scores.append({
                "id": rec["id"], "field": f, "p": float(Pf[i, j]), "p_masked": float(Pf_masked[i, j]),
                "thr": float(thr[j]), "ratio": float(Pf[i, j] / thr[j]),
                "domain_error": not rec["domain_correct"], "knn_rate": float(knn_rate[i, j]),
            })
    within = [x for x in fn_scores if not x["domain_error"]]
    ratios = np.array([x["ratio"] for x in within]) if within else np.array([])
    fn_vs_thr = {
        "n_fn": len(fn_scores),
        "n_fn_from_domain_error": len(fn_scores) - len(within),
        "within_domain": {
            "n": len(within),
            "median_p": round(float(np.median([x["p"] for x in within])), 4) if within else None,
            "median_threshold": round(float(np.median([x["thr"] for x in within])), 4) if within else None,
            "share_p_ge_0.75thr": round(float((ratios >= 0.75).mean()), 4),
            "share_p_ge_0.5thr": round(float((ratios >= 0.5).mean()), 4),
            "share_p_lt_0.1thr": round(float((ratios < 0.1).mean()), 4),
            "ratio_histogram": hist(ratios, [0, 0.1, 0.25, 0.5, 0.75, 1.0]),
            "median_knn_rate": round(float(np.median([x["knn_rate"] for x in within])), 3) if within else None,
            "share_knn_rate_0": round(float(np.mean([x["knn_rate"] == 0 for x in within])), 4) if within else None,
        },
    }
    tp_knn = [float(knn_rate[i, fidx[f]]) for i, rec in enumerate(records)
              for f in rec["gold_fields"] if f not in rec["false_negatives"]]
    fn_vs_thr["tp_median_knn_rate"] = round(float(np.median(tp_knn)), 3)

    # oracle per-field threshold on val_eval: DIAGNOSTIC ONLY (upper bound on what threshold choice could buy;
    # never applied, never saved)
    fn_vs_thr["oracle_threshold_upper_bound"] = oracle_threshold(Pf_masked, Y, thr)

    # ---- FP scores vs thresholds (correct-domain rows)
    fp_rows = [(i, f) for i, rec in enumerate(records) if rec["domain_correct"] for f in rec["false_positives"]]
    fp_ratio = np.array([Pf[i, fidx[f]] / thr[fidx[f]] for i, f in fp_rows])
    fp_vs_thr = {
        "n": len(fp_rows),
        "share_p_lt_1.5thr": round(float((fp_ratio < 1.5).mean()), 4),
        "share_p_ge_2thr": round(float((fp_ratio >= 2).mean()), 4),
        "share_knn_rate_0": round(float(np.mean([knn_rate[i, fidx[f]] == 0 for i, f in fp_rows])), 4),
        "ratio_histogram": hist(fp_ratio, [1.0, 1.25, 1.5, 2.0, 3.0]),
    }
    thr_extremes = {}
    for name, sel in (("thr>=0.5", thr >= 0.5), ("0.2<thr<0.5", (thr > 0.2) & (thr < 0.5)), ("thr<=0.2", thr <= 0.2)):
        ds = [d for d in fields if sel[fidx[d["field"]]] and d["support"] > 0]
        tp, fn_, fp_ = (sum(d[k] for d in ds) for k in ("tp", "fn", "fp"))
        p, r_, f1 = prf(tp, fp_, fn_)
        thr_extremes[name] = {"fields": len(ds), "precision": round(p, 4), "recall": round(r_, 4), "f1": round(f1, 4)}

    # ---- domain errors vs field errors
    dom_wrong = [r for r in records if not r["domain_correct"]]
    recoverable = [(i, f) for i, rec in enumerate(records) if not rec["domain_correct"]
                   for f in rec["false_negatives"] if Pf[i, fidx[f]] >= thr[fidx[f]]]
    dom_right = [r for r in records if r["domain_correct"]]
    total_fn = sum(len(r["false_negatives"]) for r in records)
    total_fp = sum(len(r["false_positives"]) for r in records)
    separation = {
        "queries": len(records),
        "domain_errors": len(dom_wrong),
        "domain_confusion": Counter(f"{r['gold_domain']} -> {r['pred_domain']}" for r in dom_wrong).most_common(),
        "fn_total": total_fn, "fp_total": total_fp,
        "fn_on_domain_errors": sum(len(r["false_negatives"]) for r in dom_wrong),
        "fn_on_domain_errors_field_head_was_right": len(recoverable),
        "fp_on_domain_errors": sum(len(r["false_positives"]) for r in dom_wrong),
        "exact_set_failures": sum(not r["exact_set"] for r in records),
        "exact_set_failures_from_domain_error": sum(not r["exact_set"] for r in dom_wrong),
        "exact_set_accuracy_given_correct_domain": round(float(np.mean([r["exact_set"] for r in dom_right])), 4),
        "error_type_given_correct_domain": Counter(
            "fn+fp" if r["false_negatives"] and r["false_positives"] else
            "fn_only" if r["false_negatives"] else "fp_only" if r["false_positives"] else "exact"
            for r in dom_right),
    }

    # ---- sparse supervision: train support vs recall / F1
    present = [d for d in fields if d["support"] > 0]
    sparse = {
        "spearman_train_support_vs_recall": round(spearman([d["train_support"] for d in present],
                                                           [d["recall"] for d in present]), 3),
        "spearman_train_support_vs_f1": round(spearman([d["train_support"] for d in present],
                                                       [d["f1"] for d in present]), 3),
        "by_train_support_bucket": support_buckets(present),
    }

    # ---- span origin (explicit wording vs derived from the item word) and operator
    origin_c = defaultdict(lambda: [0, 0])  # origin -> [missed, total]
    op_c = defaultdict(lambda: [0, 0])
    for rec in records:
        for f in rec["gold_fields"]:
            spans = rec["gold_spans"].get(short(f)) or [{}]
            o, op = spans[0].get("origin") or "?", spans[0].get("op") or "?"
            miss = f in rec["false_negatives"]
            origin_c[o][0] += miss
            origin_c[o][1] += 1
            op_c[op][0] += miss
            op_c[op][1] += 1
    recall_by = {
        "span_origin": {k: {"n": v[1], "recall": round(1 - v[0] / v[1], 4)} for k, v in origin_c.items()},
        "span_op": {k: {"n": v[1], "recall": round(1 - v[0] / v[1], 4)} for k, v in op_c.items()},
    }
    digit_fn = numeric_cue_recall(records)

    searches = [r for r in records if r["gold_domain"] != "none"]
    breakdowns = {
        "n_expected_fields": acc_table(records, "n_expected_fields"),
        "language": acc_table(records, "language"),
        "complexity": acc_table(records, "complexity"),
        "operator": acc_table(records, "operator"),
        "generation_batch_family": acc_table(records, "generation_batch_family"),
        "generation_batch": acc_table(records, "generation_batch"),
        "generator": acc_table(records, "generator"),
        "is_rephrasing": acc_table(records, "is_rephrasing"),
        "quality": acc_table(records, "quality"),
        "searches_only_language": acc_table(searches, "language"),
    }

    hashes_after = {"heads.joblib": sha(model_path), "thresholds.json": sha(thr_path)}
    assert hashes_before == hashes_after, "model artifacts changed during analysis"

    diagnostics = {
        "setup": {"partition": "val_eval", "n": len(records), "model": str(model_path.relative_to(EXP)),
                  "C": heads.C, "variant": VARIANT, "domain_calibration": heads.domain_cal.method,
                  "global_threshold": heads.field_cal.global_thr,
                  "per_field_thresholds": int((thr != heads.field_cal.global_thr).sum()),
                  "artifact_sha256_16": hashes_after, "test_split_loaded": False,
                  "generation_batch_source": str(GENERATOR_FILE) if batches else None},
        "fields": fields,
        "confusions_within_correct_domain": confusions,
        "confusions_by_field_name": [{"missed": a, "predicted_instead": b_, "count": c}
                                     for (a, b_), c in name_pair_c.most_common(15)],
        "fn_vs_threshold": fn_vs_thr,
        "domain_vs_field_errors": separation,
        "sparse_supervision": sparse,
        "recall_by": recall_by,
        "numeric_cue": digit_fn,
        "label_consistency": label_consistency(raw_train),
        "lexical_label_conflicts": lexical_conflicts(raw_train),
        "fp_vs_threshold": fp_vs_thr,
        "what_if": whatif,
        "threshold_extremes": thr_extremes,
        "exact_set_breakdowns": breakdowns,
    }
    (HERE / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=1, default=str),
                                           encoding="utf-8")
    (HERE / "diagnostics.md").write_text(render_md(diagnostics), encoding="utf-8")

    sample = draw_sample(records, Pf, thr, knn_rate, fidx, vocab)
    with open(HERE / "root_cause_sample.jsonl", "w", encoding="utf-8") as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"val_eval={len(records)} exact={np.mean([r['exact_set'] for r in records]):.3f} "
          f"FN={total_fn} FP={total_fp} domain_errors={len(dom_wrong)} sample={len(sample)}")
    print(f"artifacts unchanged: {hashes_after}")


# ------------------------------------------------------------------ pieces
def write_predictions(records, vocab):
    with open(HERE / "predictions_val_eval.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_i"}, ensure_ascii=False) + "\n")
    cols = ["id", "query", "gold_domain", "pred_domain", "domain_confidence", "domain_correct", "exact_set",
            "n_expected_fields", "gold_fields", "pred_fields", "false_negatives", "false_positives", "language",
            "complexity", "operator", "quality", "generation_batch", "generator", "is_rephrasing"]
    with open(HERE / "predictions_val_eval.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + [f"p__{v}" for v in vocab])
        for r in records:
            row = [("|".join(r[c]) if isinstance(r[c], list) else r[c]) for c in cols]
            w.writerow(row + [r["field_probabilities"][v] for v in vocab])


def hist(values, edges):
    out = {}
    for lo, hi in zip(edges, edges[1:] + [np.inf]):
        out[f"[{lo},{hi})"] = int(((values >= lo) & (values < hi)).sum())
    return out


def oracle_threshold(P, Y, thr):
    """Per-field best threshold on val_eval itself. Diagnostic upper bound only."""
    grid = np.round(np.arange(0.025, 0.976, 0.025), 3)

    def micro(pred):
        tp = (pred & Y).sum()
        return 2 * tp / max(2 * tp + (pred & ~Y).sum() + (~pred & Y).sum(), 1)

    cur = micro(P >= thr)
    best = thr.copy()
    for j in range(Y.shape[1]):
        scores = []
        for t in grid:
            trial = best.copy()
            trial[j] = t
            scores.append(micro(P >= trial))
        best[j] = grid[int(np.argmax(scores))]
    pred = P >= best
    return {"micro_f1_current": round(float(cur), 4), "micro_f1_oracle": round(float(micro(pred)), 4),
            "exact_set_current": round(float(((P >= thr) == Y).all(1).mean()), 4),
            "exact_set_oracle": round(float((pred == Y).all(1).mean()), 4),
            "note": "thresholds re-picked on val_eval itself; optimistic, not applied, not saved"}


def support_buckets(present):
    buckets = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 10**6)]
    out = {}
    for lo, hi in buckets:
        ds = [d for d in present if lo <= d["train_support"] < hi]
        if ds:
            tp, fn, fp = (sum(d[k] for d in ds) for k in ("tp", "fn", "fp"))
            p, r, f1 = prf(tp, fp, fn)
            out[f"{lo}-{hi if hi < 10**6 else '∞'}"] = {"fields": len(ds), "recall": round(r, 4),
                                                       "precision": round(p, 4), "f1": round(f1, 4)}
    return out


def span_op_stats(raw_train):
    c = defaultdict(Counter)
    for r in raw_train:
        d = r.get("domain")
        for k, spans in (r.get("spans") or {}).items():
            for s in spans[:1]:
                c[f"{d}/{k}"][s.get("op")] += 1
    return {f: dict(v) for f, v in c.items()}


def label_consistency(raw_train):
    """How the taxonomy's point / range twins are used in train: which span operators each carries.
    If a point field (מחיר) carries min/max/range spans as often as its range twin, the two are semantically
    overlapping by construction."""
    ops = span_op_stats(raw_train)
    out = []
    for f in sorted(ops):
        d, name = f.split("/", 1)
        if name.startswith("טווח_"):
            continue
        twin = f"{d}/טווח_{name}"
        alt = {"שנה": "טווח_שנה", "ק״מ": "טווח_ק״מ", "מ״ר": "טווח_מ״ר", "מס׳_חדרים": "טווח_חדרים",
               "חדרים": "טווח_חדרים", "קומה": "טווח_קומה"}.get(name)
        for t in {twin, f"{d}/{alt}" if alt else None} - {None}:
            if t in ops:
                out.append({"point_field": f, "point_ops": ops[f], "range_field": t, "range_ops": ops[t]})
    return out


TWINS = {"טווח_מחיר": "מחיר", "טווח_קומה": "קומה", "טווח_ק״מ": "ק״מ", "טווח_חדרים": "מס׳_חדרים",
         "טווח_מ״ר": "מ״ר_בנוי", "טווח_שנה": "שנה"}


def what_if(records, Pf, thr, fidx):
    """Re-score the *existing* predictions under changed assumptions (no retraining, nothing saved).
    Each number is an upper bound on what the corresponding change could buy."""
    def canon_twins(f):
        d, n = f.split("/", 1)
        return f"{d}/{TWINS.get(n, n)}"

    def canon_loc(f):
        return "יד_שנייה/עיר" if f == "יד_שנייה/אזור" else f

    def oracle_domain_pred(rec):
        if rec["domain_correct"]:
            return rec["pred_fields"]
        i = rec["_i"]
        if rec["gold_domain"] == "none":
            return []
        return [f for f, j in fidx.items() if f.startswith(rec["gold_domain"] + "/") and Pf[i, j] >= thr[j]]

    def score(pred_of, canon=lambda f: f):
        ok, tp, fp, fn = 0, 0, 0, 0
        for rec in records:
            g = {canon(f) for f in rec["gold_fields"]}
            p = {canon(f) for f in pred_of(rec)}
            ok += g == p
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
        return {"exact_set": round(ok / len(records), 4), "micro_f1": round(2 * tp / (2 * tp + fp + fn), 4)}

    base = lambda r: r["pred_fields"]  # noqa: E731
    both = lambda f: canon_loc(canon_twins(f))  # noqa: E731
    return {
        "current": score(base),
        "merge_point_range_twins": score(base, canon_twins),
        "merge_yad2_area_city": score(base, canon_loc),
        "oracle_domain": score(oracle_domain_pred),
        "twins+area_city+oracle_domain": score(oracle_domain_pred, both),
        "note": "evaluation-side what-ifs on the saved model's predictions; upper bounds, not model changes",
    }


LEXICAL_CUES = [  # (domain, description, regex on the span text) - cues seen in the root-cause sample
    ("יד_שנייה", "span starts with 'באזור'", r"^באזור"),
    ("נדל״ן", "span starts with 'באזור'", r"^באזור"),
    ("רכב", "span contains 'פרטי'", r"פרטי"),
    ("רכב", "price/km span with 'עד' (upper bound)", r"^(עד|לא יותר|מקסימום|תקרה)"),
    ("נדל״ן", "span with 'עד' (upper bound)", r"^(עד|לא יותר|מקסימום|לא מעל)"),
    ("יד_שנייה", "span mentions GB/גיגה/טרה", r"(GB|גיגה|טרה|TB)"),
    ("רכב", "span mentions 'יד'", r"\bיד\b|ביד"),
]


def lexical_conflicts(raw_train):
    """For each surface cue, which fields does the train set assign the matching span to?"""
    out = []
    for dom, desc, rx in LEXICAL_CUES:
        c = Counter()
        for r in raw_train:
            if r.get("domain") != dom:
                continue
            for k, spans in (r.get("spans") or {}).items():
                if any(re.search(rx, s["text"]) for s in spans):
                    c[k] += 1
        out.append({"domain": dom, "cue": desc, "fields": dict(c.most_common(6))})
    return out


def numeric_cue_recall(records):
    """Recall of fields whose gold span text contains a digit vs spans without digits."""
    c = {"digit": [0, 0], "no_digit": [0, 0]}
    for rec in records:
        for f in rec["gold_fields"]:
            spans = rec["gold_spans"].get(short(f)) or [{"text": ""}]
            k = "digit" if re.search(r"\d", spans[0]["text"]) else "no_digit"
            c[k][1] += 1
            c[k][0] += f in rec["false_negatives"]
    return {k: {"n": v[1], "recall": round(1 - v[0] / v[1], 4) if v[1] else None} for k, v in c.items()}


def draw_sample(records, Pf, thr, knn_rate, fidx, vocab, n_fn=40, n_fp=30, n_dom=10, n_ctrl=20):
    """~100 cases: within-domain FN (top FN fields first, mixing near and far misses), FP, domain errors,
    correct controls (half hard: compound/ambiguous with >=3 fields)."""
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = {r["id"]: i for i, r in enumerate(records)}
    used, out = set(), []

    def view(rec, reason, focus):
        i = idx[rec["id"]]
        interesting = set(rec["gold_fields"]) | set(rec["pred_fields"])
        top_other = sorted(((float(Pf[i, j]), v) for j, v in enumerate(vocab) if v not in interesting),
                           reverse=True)[:3]
        return {
            "case": len(out) + 1, "reason": reason, "focus_field": focus, "id": rec["id"], "query": rec["query"],
            "gold_domain": rec["gold_domain"], "pred_domain": rec["pred_domain"],
            "domain_confidence": rec["domain_confidence"], "gold_params": rec["gold_params"],
            "gold_spans": rec["gold_spans"], "pred_fields": rec["pred_fields"],
            "false_negatives": rec["false_negatives"], "false_positives": rec["false_positives"],
            "scores": {f: {"p": round(float(Pf[i, fidx[f]]), 3), "thr": round(float(thr[fidx[f]]), 3),
                           "knn10": round(float(knn_rate[i, fidx[f]]), 1)} for f in sorted(interesting)},
            "top_other": [{"field": v, "p": round(p, 3)} for p, v in top_other],
            "language": rec["language"], "complexity": rec["complexity"], "operator": rec["operator"],
            "generation_batch": rec["generation_batch"], "generator": rec["generator"],
        }

    def take(pool, k, reason, focus_of):
        pool = [r for r in pool if r["id"] not in used]
        for j in rng.permutation(len(pool))[:k]:
            rec = pool[j]
            used.add(rec["id"])
            out.append(view(rec, reason, focus_of(rec)))

    ok_dom = [r for r in records if r["domain_correct"]]
    # FN: round-robin over the most-missed fields so the sample covers them rather than one dominant field
    fn_fields = Counter(f for r in ok_dom for f in r["false_negatives"]).most_common()
    per = defaultdict(list)
    for r in ok_dom:
        for f in r["false_negatives"]:
            per[f].append(r)
    taken = 0
    while taken < n_fn and any(per.values()):
        for f, _ in fn_fields:
            cand = [r for r in per[f] if r["id"] not in used]
            if cand and taken < n_fn:
                rec = cand[rng.integers(len(cand))]
                used.add(rec["id"])
                out.append(view(rec, "false_negative", f))
                taken += 1
            per[f] = [r for r in per[f] if r["id"] not in used]
    take([r for r in ok_dom if r["false_positives"]], n_fp, "false_positive", lambda r: r["false_positives"][0])
    take([r for r in records if not r["domain_correct"]], n_dom, "domain_error", lambda r: None)
    hard = [r for r in ok_dom if r["exact_set"] and r["n_expected_fields"] >= 3
            and r["complexity"] in ("compound", "ambiguous")]
    take(hard, n_ctrl // 2, "control_correct_hard", lambda r: None)
    take([r for r in ok_dom if r["exact_set"]], n_ctrl - n_ctrl // 2, "control_correct", lambda r: None)
    return out


def render_md(d):
    L = ["# Experiment 1 - field classifier diagnostics (val_eval, C=64, saved model)", ""]
    s = d["setup"]
    L += [f"n={s['n']} · variant `{s['variant']}` · global thr {s['global_threshold']} · "
          f"{s['per_field_thresholds']} per-field thresholds · artifacts sha {s['artifact_sha256_16']} · "
          f"test loaded: {s['test_split_loaded']}", ""]
    L += ["## What-if re-scoring of the saved model's predictions (upper bounds, no retraining)", "",
          "| scenario | exact set | micro-F1 |", "|---|---|---|"]
    L += [f"| {k} | {v['exact_set']:.3f} | {v['micro_f1']:.3f} |" for k, v in d["what_if"].items() if k != "note"]
    L.append("")
    sep = d["domain_vs_field_errors"]
    L += ["## Domain errors vs field errors", "",
          f"- domain errors: {sep['domain_errors']} / {sep['queries']} ({sep['domain_confusion']})",
          f"- FN: {sep['fn_total']} total, {sep['fn_on_domain_errors']} on domain-error rows "
          f"({sep['fn_on_domain_errors_field_head_was_right']} of those were above threshold before the domain mask)",
          f"- FP: {sep['fp_total']} total, {sep['fp_on_domain_errors']} on domain-error rows",
          f"- exact-set failures: {sep['exact_set_failures']}, of which {sep['exact_set_failures_from_domain_error']}"
          f" on domain-error rows; exact-set acc given correct domain = "
          f"{sep['exact_set_accuracy_given_correct_domain']}",
          f"- error type given correct domain: {dict(sep['error_type_given_correct_domain'])}", ""]
    L += ["## Fields ranked by errors (FN + FP)", "",
          "| field | FN | FP | precision | recall | F1 | support | train support | thr | Platt |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in d["fields"][:30]:
        L.append(f"| {x['field']} | {x['fn']} | {x['fp']} | {x['precision']:.2f} | {x['recall']:.2f} | "
                 f"{x['f1']:.2f} | {x['support']} | {x['train_support']} | {x['threshold']} | "
                 f"{'y' if x['platt_calibrated'] else 'n'} |")
    L += ["", "## Most common confusions (correct domain; missed → predicted instead, same query)", "",
          "| missed | predicted instead | count |", "|---|---|---|"]
    L += [f"| {c['missed']} | {c['predicted_instead']} | {c['count']} |" for c in d["confusions_within_correct_domain"]]
    f = d["fn_vs_threshold"]
    w = f["within_domain"]
    L += ["", "## False-negative scores vs thresholds (correct-domain rows)", "",
          f"- FN={f['n_fn']} ({f['n_fn_from_domain_error']} from domain errors); within-domain n={w['n']}",
          f"- median p={w['median_p']} vs median threshold {w['median_threshold']}",
          f"- p ≥ 0.75·thr: {w['share_p_ge_0.75thr']:.1%} · p ≥ 0.5·thr: {w['share_p_ge_0.5thr']:.1%} · "
          f"p < 0.1·thr: {w['share_p_lt_0.1thr']:.1%}",
          f"- p/thr histogram: {w['ratio_histogram']}",
          f"- kNN-10 share of neighbours carrying the field: FN median {w['median_knn_rate']} "
          f"(0 for {w['share_knn_rate_0']:.1%}) vs TP median {f['tp_median_knn_rate']}",
          f"- oracle per-field thresholds on val_eval (upper bound, not applied): {f['oracle_threshold_upper_bound']}",
          ""]
    fv = d["fp_vs_threshold"]
    L += ["## False-positive scores vs thresholds (correct-domain rows)", "",
          f"- n={fv['n']} · p < 1.5·thr: {fv['share_p_lt_1.5thr']:.1%} · p ≥ 2·thr: {fv['share_p_ge_2thr']:.1%} · "
          f"no kNN-10 neighbour carries the field: {fv['share_knn_rate_0']:.1%}",
          f"- p/thr histogram: {fv['ratio_histogram']}", "",
          "| threshold band | fields | precision | recall | F1 |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v['fields']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |"
          for k, v in d["threshold_extremes"].items()]
    L += ["", "## Lexical cues and the fields train assigns them to", "", "| domain | cue | fields (train span count) |",
          "|---|---|---|"]
    L += [f"| {x['domain']} | {x['cue']} | {x['fields']} |" for x in d["lexical_label_conflicts"]]
    L.append("")
    sp = d["sparse_supervision"]
    L += ["## Sparse supervision", "",
          f"- Spearman(train support, recall) = {sp['spearman_train_support_vs_recall']}, "
          f"(train support, F1) = {sp['spearman_train_support_vs_f1']}", "",
          "| train support | fields | recall | precision | F1 |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v['fields']} | {v['recall']:.3f} | {v['precision']:.3f} | {v['f1']:.3f} |"
          for k, v in sp["by_train_support_bucket"].items()]
    rb = d["recall_by"]
    L += ["", "## Recall by gold span origin / operator / numeric cue", "",
          f"- origin: {rb['span_origin']}", f"- op: {rb['span_op']}", f"- span has digit: {d['numeric_cue']}", ""]
    L += ["## Point vs range twins: span operators in train", "", "| point field | ops | range field | ops |",
          "|---|---|---|---|"]
    L += [f"| {x['point_field']} | {x['point_ops']} | {x['range_field']} | {x['range_ops']} |"
          for x in d["label_consistency"]]
    L += ["", "## Exact field-set accuracy breakdowns", ""]
    for k, tab in d["exact_set_breakdowns"].items():
        L += [f"### {k}", "", "| value | n | exact set | domain acc | FN/query | FP/query |", "|---|---|---|---|---|---|"]
        L += [f"| {v} | {x['n']} | {x['exact_set_accuracy']:.3f} | {x['domain_accuracy']:.3f} | "
              f"{x['fn_per_query']} | {x['fp_per_query']} |" for v, x in tab.items()]
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
