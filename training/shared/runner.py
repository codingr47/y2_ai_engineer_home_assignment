"""Generic experiment driver: features x label space -> sweep / learning curve / final.

Data roles (never changed):
  train      fits models and feature extractors (TF-IDF vocabularies are fit on the run's training subset)
  val_eval   selects hyperparameters (C, w) and compares representations -> DEVELOPMENT metrics
  val_calib  probability calibration + field thresholds only (inside Heads.calibrate)
  test       scored once per experiment for the frozen selected config (--final-test)

Primary metric: Slot Exact-Set Accuracy (70 slots; independent of the point/range resolver).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from . import paths
from .features import make_featurizer
from .metadata import generation_batches, raw_rows
from .operators import detect_operator, is_bounded
from .slots import fields_to_slots, is_twin_slot, slot_names, slot_vocab
from .taxonomy_rules import Resolver, close_hierarchy

from train import Heads, Split, load_split, load_vocab  # noqa: E402  (experiment_1, see paths.py)
from evaluate import domain_metrics, field_metrics  # noqa: E402
from run_experiment import family_subsample  # noqa: E402

C_GRID = (1, 4, 16, 64, 256)
W_GRID = (0.25, 0.5, 1, 2, 4)
TRAIN_SIZES = (250, 500, 1000, 2000, 4000, None)
CURVE_SEEDS = (0, 1, 2)
COVERAGES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass
class Spec:
    name: str              # e.g. "exp2"
    kind: str              # "e5" | "tfidf" | "e5+tfidf"
    label_space: str       # "fields" (77) | "slots" (70)
    out_dir: Path
    description: str = ""


@dataclass
class Bundle:
    """Everything needed at inference: featurizer -> heads -> taxonomy post-processing."""
    spec_name: str
    kind: str
    label_space: str
    C: float
    w: float | None
    featurizer: object
    heads: Heads
    resolver: Resolver
    vocab77: list[str] = field(default_factory=list)

    @property
    def vocab(self):
        return self.heads.vocab

    def predict_X(self, X):
        Pd, Pf, pred = self.heads.predict(X, "full")
        return Pd, Pf, pred

    def postprocess(self, pred_row, local_spans=None):
        """Predicted labels -> slots -> taxonomy field names (operator-free default unless spans are given)."""
        labels = [self.vocab[j] for j in np.flatnonzero(pred_row)]
        slots = labels if self.label_space == "slots" else fields_to_slots(labels, self.vocab77)
        names = [self.resolver.resolve(s, (local_spans or {}).get(s),
                                       detect_operator((local_spans or {})[s]) if (local_spans or {}).get(s) else None)
                 for s in slots] if self.label_space == "slots" else labels
        return slots, names


# ------------------------------------------------------------------ data
_BASE: dict[str, Split] = {}


def base_split(name: str) -> Split:
    """experiment_1's labels/meta + cached E5 vectors (labels are the original 77 fields)."""
    if name not in _BASE:
        _BASE[name] = load_split(name)
    return _BASE[name]


def relabel(split: Split, X, label_space: str, vocab77: list[str]) -> Split:
    if label_space == "fields":
        labels = split.labels
    else:
        labels = [{**r, "field_labels": fields_to_slots(r["field_labels"], vocab77)} for r in split.labels]
    return Split(split.name, X, labels, split.meta)


# ------------------------------------------------------------------ train
def fit(spec: Spec, C: float, w: float | None, size=None, seed=0) -> Bundle:
    vocab77 = load_vocab()
    train = base_split("train")
    sub = train.subset(family_subsample(train, size, seed))
    feat = make_featurizer(spec.kind, w or 1.0).fit(sub)
    vocab = vocab77 if spec.label_space == "fields" else slot_vocab(vocab77)
    tr = relabel(sub, feat.transform(sub), spec.label_space, vocab77)
    calib = base_split("val_calib")
    ca = relabel(calib, feat.transform(calib), spec.label_space, vocab77)
    heads = Heads(vocab=vocab, C=C).fit(tr).calibrate(ca)
    resolver = Resolver.from_train(sub.fields, slot_vocab(vocab77))
    return Bundle(spec.name, spec.kind, spec.label_space, C, w, feat, heads, resolver, vocab77)


# ------------------------------------------------------------------ scoring
def _binary(rows, vocab):
    idx = {v: j for j, v in enumerate(vocab)}
    Y = np.zeros((len(rows), len(vocab)), dtype=bool)
    for i, r in enumerate(rows):
        for f in r:
            Y[i, idx[f]] = True
    return Y


def _gold_spans(split_name: str, ids):
    raw = raw_rows(split_name)
    out = []
    for i in ids:
        r = raw[i]
        d = r.get("domain")
        out.append({f"{d}/{k}": v[0] for k, v in (r.get("spans") or {}).items()} if d else {})
    return out


def score(b: Bundle, split_name: str, records: bool = False) -> dict:
    split = base_split(split_name)
    X = b.featurizer.transform(split)
    t0 = time.perf_counter()
    Pd, Pf, pred = b.predict_X(X)
    classify_s = time.perf_counter() - t0
    classes = b.heads.domain_model.classes_
    vocab, vocab77 = b.vocab, b.vocab77
    svocab = slot_vocab(vocab77)
    pred_dom = classes[Pd.argmax(1)]
    gold77 = split.fields
    gold_slots = [fields_to_slots(f, vocab77) for f in gold77]
    ids = [r["id"] for r in split.labels]
    spans = _gold_spans(split_name, ids)

    pred_labels = [[vocab[j] for j in np.flatnonzero(p)] for p in pred]
    pred_slots = pred_labels if b.label_space == "slots" else [fields_to_slots(p, vocab77) for p in pred_labels]

    # strict names: deployable (no operator), provisional (detector on gold local span), oracle (gold op)
    def names_for(i, mode):
        if b.label_space == "fields":
            return pred_labels[i]
        out = []
        for s in pred_slots[i]:
            op = None
            if is_twin_slot(s) and mode != "deployable":
                sp = next((spans[i][f] for f in slot_names(s) if f in spans[i]), None)
                if sp is not None:
                    op = sp["op"] if mode == "oracle" else detect_operator(sp["text"])
            out.append(b.resolver.resolve(s, None, op))
        return out

    strict = {m: [names_for(i, m) for i in range(len(ids))] for m in ("deployable", "provisional", "oracle")}
    rules_slots = [close_hierarchy(p) for p in pred_slots]

    Ys, Ps = _binary(gold_slots, svocab), _binary(pred_slots, svocab)
    fm = field_metrics(Ys, Ps, svocab)
    dm = domain_metrics(split.domains, Pd, classes)
    Y77 = _binary(gold77, vocab77)
    strict_m = {m: field_metrics(Y77, _binary(v, vocab77), vocab77) for m, v in strict.items()}
    rules_exact = float(np.mean([set(r) == set(g) for r, g in zip(rules_slots, gold_slots)]))

    res = {
        "split": split_name, "development": split_name == "val_eval", "n": len(ids),
        "domain_accuracy": dm["accuracy"], "domain_ece": dm["ece"],
        "slot_micro_f1": fm["micro_f1"], "slot_macro_f1": fm["macro_f1"], "slot_exact_set": fm["exact_set_accuracy"],
        "slot_exact_set_rules": rules_exact,
        "strict_exact_set": {m: v["exact_set_accuracy"] for m, v in strict_m.items()},
        "strict_micro_f1": {m: v["micro_f1"] for m, v in strict_m.items()},
        "per_slot": fm["per_field"],
        "classify_seconds": round(classify_s, 4),
    }
    exact = np.array([set(p) == set(g) for p, g in zip(pred_slots, gold_slots)])
    conf = confidence_features(b, Pd, Pf, pred, classes)
    res["confidence"] = confidence_quality(conf, exact)
    if records:
        res["records"] = make_records(b, split, ids, Pd, Pf, pred_slots, strict, conf, exact, classes)
    return res


def confidence_features(b, Pd, Pf, pred, classes):
    thr = b.heads.field_cal.thr
    out = []
    for i in range(len(Pd)):
        p, sel = Pf[i], pred[i]
        ratio = p / thr
        d = np.sort(Pd[i])[::-1]
        mins = float(p[sel].min()) if sel.any() else None
        maxr = float(p[~sel].max()) if (~sel).any() else 0.0
        minr = float(ratio[sel].min()) if sel.any() else None
        maxrr = float(ratio[~sel].max()) if (~sel).any() else 0.0
        out.append({
            "domain_confidence": float(d[0]), "domain_margin": float(d[0] - d[1]),
            "min_selected_probability": mins, "max_rejected_probability": maxr,
            # no slot selected (e.g. domain 'none'): the margin uses 1.0 in place of min_selected
            "prediction_margin": (mins if mins is not None else 1.0) - maxr,
            "min_selected_ratio": minr, "max_rejected_ratio": maxrr,
            "prediction_margin_ratio": (minr if minr is not None else 1.0) - maxrr,
            "n_selected": int(sel.sum()),
        })
    return out


def confidence_quality(conf, exact):
    """How well each inference-time score ranks queries by slot_exact_match (descriptive; no router)."""
    out = {}
    for key in ("domain_confidence", "domain_margin", "prediction_margin", "prediction_margin_ratio",
                "max_rejected_probability", "min_selected_probability"):
        s = np.array([c[key] if c[key] is not None else 1.0 for c in conf], dtype=float)
        if key == "max_rejected_probability":
            s = -s  # lower rejected probability = more confident
        auroc = float(roc_auc_score(exact, s)) if 0 < exact.mean() < 1 else None
        order = np.argsort(-s, kind="stable")
        rc = []
        for cov in COVERAGES:
            k = max(1, int(round(cov * len(s))))
            rc.append({"coverage": cov, "slot_exact_set": round(float(exact[order[:k]].mean()), 4)})
        out[key] = {"auroc": round(auroc, 4) if auroc is not None else None, "risk_coverage": rc}
    return out


def make_records(b, split, ids, Pd, Pf, pred_slots, strict, conf, exact, classes):
    batches = generation_batches()
    raw = raw_rows(split.name)
    recs = []
    for i, rid in enumerate(ids):
        r, m = raw[rid], split.meta[i]
        md = r.get("metadata") or {}
        bt = batches.get(rid, {})
        recs.append({
            "id": rid, "query": m["query"],
            "gold_domain": split.labels[i]["domain_label"],
            "pred_domain": str(classes[Pd[i].argmax()]),
            "domain_probs": {str(c): round(float(p), 5) for c, p in zip(classes, Pd[i])},
            "slot_probs": {v: round(float(p), 5) for v, p in zip(b.vocab, Pf[i])},
            "predicted_slots": pred_slots[i],
            "predicted_fields": {k: v[i] for k, v in strict.items()},
            "gold_fields": split.labels[i]["field_labels"],
            **conf[i],
            # evaluation target computed from ground truth - NEVER an inference-time feature
            "slot_exact_match": bool(exact[i]),
            "language": (md.get("language") or [None])[0], "complexity": (md.get("complexity") or [None])[0],
            "operator": md.get("operator") or [], "n_tokens": len(m["query"].split()),
            "family": m.get("family"), "is_rephrasing": m.get("family") not in (None, rid),
            "generation_batch": bt.get("batch"), "generator": bt.get("generator"), "quality": r.get("quality"),
        })
    return recs


def resolver_diagnostics(b: Bundle, split_name: str, quality: str | None = None) -> dict:
    """Independent of the classifier: operator detector vs gold op on gold twin spans, and which field name each
    resolver mode produces vs the gold name. Also checks mixed-operator queries (exact + bounded in one query).
    `quality` restricts to 'gold' or 'silver' rows."""
    raw = raw_rows(split_name)
    split = base_split(split_name)
    svocab = slot_vocab(b.vocab77)
    twin_of = {f: s for s in svocab if is_twin_slot(s) for f in slot_names(s)}
    n = 0
    op_ok = bounded_ok = 0
    key = {"deployable": 0, "provisional": 0, "oracle": 0}
    confusion = {}
    mixed_rows = mixed_ok = 0
    for lab in split.labels:
        r = raw[lab["id"]]
        if quality is not None and r.get("quality") != quality:
            continue
        d = r.get("domain")
        sp = {f"{d}/{k}": v[0] for k, v in (r.get("spans") or {}).items()} if d else {}
        numeric = [s for s in sp.values() if s["op"] in ("exact", "min", "max", "range")]
        ops = {is_bounded(s["op"]) for s in numeric}
        if len(numeric) >= 2 and ops == {True, False}:
            mixed_rows += 1
            mixed_ok += all(is_bounded(detect_operator(s["text"])) == is_bounded(s["op"]) for s in numeric)
        for f, s in sp.items():
            if f not in twin_of:
                continue
            n += 1
            det = detect_operator(s["text"])
            op_ok += det == s["op"]
            bounded_ok += is_bounded(det) == is_bounded(s["op"])
            confusion.setdefault(s["op"], {}).setdefault(det, 0)
            confusion[s["op"]][det] += 1
            slot = twin_of[f]
            key["deployable"] += b.resolver.resolve(slot) == f
            key["provisional"] += b.resolver.resolve(slot, s["text"], det) == f
            key["oracle"] += b.resolver.resolve(slot, s["text"], s["op"]) == f
    return {
        "split": split_name, "quality": quality or "all", "twin_spans": n,
        "detector_accuracy_4way": round(op_ok / n, 4), "detector_accuracy_bounded_vs_exact": round(bounded_ok / n, 4),
        "detector_confusion": confusion,
        "field_name_accuracy": {k: round(v / n, 4) for k, v in key.items()},
        "mixed_operator_queries": mixed_rows,
        "mixed_operator_queries_all_fields_detected_right": mixed_ok,
        "default_names": b.resolver.default_names,
        "note": "oracle = gold operator + taxonomy rule; its gap to 1.0 is disagreement between the taxonomy rule and "
                "the gold field names (label noise), not a model error",
    }


SUMMARY_KEYS = ("domain_accuracy", "slot_micro_f1", "slot_macro_f1", "slot_exact_set", "slot_exact_set_rules",
                "domain_ece")


def summarize(res):
    out = {k: round(res[k], 4) for k in SUMMARY_KEYS}
    out.update({f"strict_exact_{m}": round(v, 4) for m, v in res["strict_exact_set"].items()})
    return out


# ------------------------------------------------------------------ modes
def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def sweep(spec: Spec, c_grid=C_GRID, w_grid=None):
    rows = []
    for C in c_grid:
        for w in (w_grid or (None,)):
            b = fit(spec, C, w)
            s = summarize(score(b, "val_eval"))
            rows.append({"C": C, "w": w, **s})
            print(f"[{spec.name}] C={C} w={w} " + " ".join(f"{k}={v:.3f}" for k, v in s.items()), flush=True)
    best = max(rows, key=lambda r: (r["slot_exact_set"], r["slot_micro_f1"]))
    sel = {"C": best["C"], "w": best["w"], "selected_on": "val_eval slot_exact_set (development)",
           "rule": "max slot exact-set, tie -> slot micro-F1"}
    _dump(spec.out_dir / "results" / "sweep.json", {"rows": rows, "selected": sel})
    _dump(spec.out_dir / "results" / "selected.json", sel)
    cols = ["C", "w", *SUMMARY_KEYS, "strict_exact_deployable"]
    md = [f"# {spec.name} sweep (val_eval, development)", "", "| " + " | ".join(cols) + " |",
          "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |"
           for r in rows]
    md += ["", f"Selected: C={sel['C']} w={sel['w']} ({sel['rule']})"]
    (spec.out_dir / "results" / "sweep.md").write_text("\n".join(md), encoding="utf-8")
    return sel


def selected(spec: Spec):
    return json.loads((spec.out_dir / "results" / "selected.json").read_text(encoding="utf-8"))


def curve(spec: Spec):
    sel = selected(spec)
    pts = []
    for size in TRAIN_SIZES:
        runs = []
        for seed in (CURVE_SEEDS if size else (0,)):
            b = fit(spec, sel["C"], sel["w"], size, seed)
            n = len(family_subsample(base_split("train"), size, seed))
            runs.append({"seed": seed, "size": n, **summarize(score(b, "val_eval"))})
        pts.append({"target_size": size or "all", "runs": runs})
        m = np.mean([r["slot_exact_set"] for r in runs])
        print(f"[{spec.name}] size={size or 'all'} slot_exact={m:.3f}", flush=True)
    _dump(spec.out_dir / "results" / "learning_curve.json", {"selected": sel, "points": pts})
    return pts


def final(spec: Spec, with_test: bool):
    sel = selected(spec)
    b = fit(spec, sel["C"], sel["w"])
    models = spec.out_dir / "artifacts" / "models"
    models.mkdir(parents=True, exist_ok=True)
    joblib.dump(b, models / "bundle.joblib")
    out = {"spec": {"name": spec.name, "kind": spec.kind, "label_space": spec.label_space,
                    "description": spec.description}, "selected": sel}
    for name in ("val_calib", "val_eval") + (("test",) if with_test else ()):
        res = score(b, name, records=True)
        recs = res.pop("records")
        with open(spec.out_dir / "results" / f"predictions_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out[name] = res
        print(f"[{spec.name}] {name}: " + " ".join(f"{k}={v:.3f}" for k, v in summarize(res).items()), flush=True)
    out["val_calib"]["note"] = "in-sample for calibration and thresholds"
    out["resolver_val_eval"] = resolver_diagnostics(b, "val_eval")
    out["thresholds"] = dict(zip(b.vocab, map(float, b.heads.field_cal.thr)))
    out["n_features"] = getattr(b.featurizer, "n_features", None) or getattr(
        getattr(b.featurizer, "tfidf", None), "n_features", None)
    _dump(spec.out_dir / "results" / ("final_test.json" if with_test else "final_dev.json"), out)
    return out


def regression_check():
    """E5 + 77 fields through this runner must reproduce experiment 1 (val_eval exact 0.602, micro-F1 0.881)."""
    spec = Spec("exp1-regression", "e5", "fields", paths.TRAINING / "experiment_1_slots" / "regression")
    res = score(fit(spec, 64, None), "val_eval")
    got = (round(res["strict_exact_set"]["deployable"], 3), round(res["strict_micro_f1"]["deployable"], 3))
    print(f"regression: strict exact={got[0]} micro-F1={got[1]} (expected 0.602 / 0.881)")
    assert got == (0.602, 0.881), got
    return got


def cli(spec: Spec, w_grid=None):
    ap = argparse.ArgumentParser(description=spec.description)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--final", action="store_true", help="fit selected config, write dev predictions")
    ap.add_argument("--final-test", action="store_true", help="as --final, plus the one-time test evaluation")
    ap.add_argument("--regression", action="store_true")
    a = ap.parse_args()
    if a.regression:
        regression_check()
    if a.sweep:
        sweep(spec, w_grid=w_grid)
    if a.curve:
        curve(spec)
    if a.final or a.final_test:
        final(spec, with_test=a.final_test)
