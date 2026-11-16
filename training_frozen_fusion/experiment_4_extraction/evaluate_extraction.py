"""Development evaluation of the field-local value extractor in three scenarios (val_eval = development).

    python training_frozen_fusion/experiment_4_extraction/evaluate_extraction.py                 # lexicon + hybrid
    python training_frozen_fusion/experiment_4_extraction/evaluate_extraction.py --tune-tagger   # tune tagger threshold on val_calib first

Scenario 1  gold fields supplied     : extraction quality independent of classification
Scenario 2  predicted fields supplied: the real pipeline (frozen Exp3 v2 classifier)
Scenario 3  complete JSON            : {category, params} exact, with a waterfall and counterfactuals

Labels: schema v2 (point fields carry bounds), applied at load time. The classifier is frozen; nothing here is fit
on val_eval. Test is not scored in this iteration (--final-test exists for the later one-time run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import paths  # noqa: E402
from shared.dataset import load_rows  # noqa: E402
from shared.runner import base_split  # noqa: E402

from src.extraction.extractor import ARTIFACTS, FieldExtractor  # noqa: E402
from src.extraction.types import values_equal  # noqa: E402

RESULTS = HERE / "results"
BUNDLE = paths.TRAINING / "experiment_3" / "artifacts" / "models_v2" / "bundle.joblib"
CATEGORY_ORDER = ["no_span", "wrong_span", "operator", "number_parse", "operator+number", "enum_normalization",
                  "derived_value", "list_partial", "free_text_normalization", "date", "boolean_polarity"]
CATEGORY_GROUP = {  # the user's four questions
    "no_span": "span detection", "wrong_span": "span detection",
    "operator": "numeric interpretation", "number_parse": "numeric interpretation",
    "operator+number": "numeric interpretation",
    "enum_normalization": "enum normalization", "derived_value": "enum normalization",
    "list_partial": "enum normalization",
    "free_text_normalization": "other extraction", "date": "other extraction", "boolean_polarity": "other extraction",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------------ classification (frozen)
def classify(split: str):
    """Frozen Exp3 v2 predictions: id -> (domain, [fields])."""
    b = joblib.load(BUNDLE)
    s = base_split(split)
    Pd, _, pred = b.predict_X(b.featurizer.transform(s))
    classes = b.heads.domain_model.classes_
    return {lab["id"]: (str(classes[Pd[i].argmax()]), [b.vocab[j] for j in np.flatnonzero(pred[i])])
            for i, lab in enumerate(s.labels)}


# ------------------------------------------------------------------ failure categories
def _numbers(v):
    if isinstance(v, dict):
        return sorted(float(x) for x in v.values())
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [float(v)]
    return []


def _gold_op(gold_value):
    if isinstance(gold_value, dict):
        return "range" if len(gold_value) == 2 else next(iter(gold_value))
    return "exact"


def categorize(ftype: str, gold_value, gold_spans: list[dict], e) -> str:
    if not e.found:
        return "no_span"
    if e.start is not None and gold_spans and not any(e.start < s["end"] and e.end > s["start"] for s in gold_spans):
        return "wrong_span"
    if ftype == "numeric":
        op_ok = (e.operator or "exact") == _gold_op(gold_value)
        num_ok = _numbers(e.value) == _numbers(gold_value)
        return "operator" if num_ok and not op_ok else "number_parse" if op_ok else "operator+number"
    if ftype == "bool":
        return "boolean_polarity"
    if ftype == "date":
        return "date"
    if ftype == "text":
        return "free_text_normalization"
    if isinstance(gold_value, list) or isinstance(e.value, list):
        gl = set(map(str, gold_value if isinstance(gold_value, list) else [gold_value]))
        pl = set(map(str, e.value if isinstance(e.value, list) else [e.value]))
        if gl & pl:
            return "list_partial"
    if gold_spans and gold_spans[0].get("origin") == "derived":
        return "derived_value"
    return "enum_normalization"


def span_agreement(gold_spans, e) -> str:
    if not e.found or e.start is None or not gold_spans:
        return "none"
    if any(e.start == s["start"] and e.end == s["end"] for s in gold_spans):
        return "exact"
    return "overlap" if any(e.start < s["end"] and e.end > s["start"] for s in gold_spans) else "disjoint"


# ------------------------------------------------------------------ scenarios
def scenario1(ex: FieldExtractor, rows):
    recs = []
    for r in rows:
        d = r.get("domain")
        params = (r.get("expected") or {}).get("params") or {}
        if not d or not params:
            continue
        out = ex.extract_all(r["query"], d, [f"{d}/{k}" for k in params])
        for k, gv in params.items():
            f = f"{d}/{k}"
            e = out[f]
            fi = ex.model.fields.get(f)
            ftype = fi.type if fi else "?"
            spans = (r.get("spans") or {}).get(k, [])
            ok = values_equal(e.value, gv)
            recs.append({"id": r["id"], "field": f, "type": ftype, "gold": gv, "pred": e.value,
                         "gold_op": _gold_op(gv) if ftype == "numeric" else None, "pred_op": e.operator,
                         "correct": ok, "category": None if ok else categorize(ftype, gv, spans, e),
                         "span": span_agreement(spans, e), "source": e.source, "reason": e.reason,
                         "origin": spans[0].get("origin") if spans else None,
                         "language": ((r.get("metadata") or {}).get("language") or [None])[0],
                         "pred_text": e.text, "gold_text": spans[0]["text"] if spans else None})
    return recs


def summarize_s1(recs, rows):
    n = len(recs)
    ok = sum(r["correct"] for r in recs)
    by = lambda key: {k: {"n": len(v), "accuracy": round(sum(x["correct"] for x in v) / len(v), 4)}  # noqa: E731
                      for k, v in sorted(_group(recs, key).items(), key=lambda kv: str(kv[0]))}
    cats = Counter(r["category"] for r in recs if not r["correct"])
    numeric = [r for r in recs if r["type"] == "numeric"]
    row_ok = defaultdict(lambda: True)
    for r in recs:
        row_ok[r["id"]] &= r["correct"]
    searches = [r for r in rows if r.get("domain") and (r.get("expected") or {}).get("params")]
    return {
        "value_accuracy": round(ok / n, 4), "n_values": n,
        "row_all_values_correct": round(sum(row_ok[r["id"]] for r in searches) / len(searches), 4),
        "n_rows": len(searches),
        "operator_accuracy_numeric": round(np.mean([r["pred_op"] == r["gold_op"] for r in numeric]), 4)
        if numeric else None,
        "span_agreement": dict(Counter(r["span"] for r in recs)),
        "failure_categories": {c: cats.get(c, 0) for c in CATEGORY_ORDER},
        "failure_groups": dict(Counter(CATEGORY_GROUP[c] for c in cats.elements())),
        "by_type": by("type"), "by_origin": by("origin"), "by_language": by("language"), "by_source": by("source"),
        "by_field": by("field"),
    }


def _group(recs, key):
    g = defaultdict(list)
    for r in recs:
        g[r[key]].append(r)
    return g


def scenario2_3(ex: FieldExtractor, rows, preds):
    pair_recs, row_recs = [], []
    for r in rows:
        gd = r.get("domain") or "none"
        gparams = (r.get("expected") or {}).get("params") or {}
        gfields = {f"{gd}/{k}" for k in gparams} if gd != "none" else set()
        pd, pf = preds[r["id"]]
        pfields = set(pf) if pd != "none" else set()
        out = ex.extract_all(r["query"], pd, sorted(pfields)) if pd != "none" else {}
        domain_ok = pd == gd
        fields_ok = domain_ok and pfields == gfields
        cats = []
        for f in sorted(pfields | gfields):
            k = f.split("/", 1)[1]
            if f in pfields and f in gfields:
                e = out[f]
                ok = values_equal(e.value, gparams[k])
                fi = ex.model.fields.get(f)
                cat = None if ok else categorize(fi.type if fi else "?", gparams[k],
                                                 (r.get("spans") or {}).get(k, []), e)
                kind = "correct" if ok else "wrong_value"
                if cat:
                    cats.append(cat)
            elif f in pfields:
                kind, cat = ("spurious_dropped" if not out[f].found else "spurious"), None
            else:
                kind, cat = "missed", None
            pair_recs.append({"id": r["id"], "field": f, "kind": kind, "category": cat,
                              "domain_error": not domain_ok})
        pred_json = FieldExtractor.to_json(pd, out)
        gold_json = None if gd == "none" else {"category": gd, "params": gparams}
        json_ok = _json_equal(pred_json, gold_json)
        stage = "json_exact" if json_ok else ("domain" if not domain_ok else "field_set" if not fields_ok
                                              else "values")
        primary = next((c for c in CATEGORY_ORDER if c in cats), None) if stage == "values" else None
        row_recs.append({"id": r["id"], "json_exact": json_ok, "domain_ok": domain_ok, "field_set_ok": fields_ok,
                         "failed_stage": stage, "value_categories": cats, "primary_category": primary,
                         "rescued": json_ok and not fields_ok,
                         "language": ((r.get("metadata") or {}).get("language") or [None])[0]})
    return pair_recs, row_recs


def _json_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a["category"] == b["category"] and set(a["params"]) == set(b["params"]) and \
        all(values_equal(a["params"][k], b["params"][k]) for k in a["params"])


def summarize_s2(pairs):
    k = Counter(p["kind"] for p in pairs)
    tp = k["correct"]
    predicted = tp + k["wrong_value"] + k["spurious"]          # pairs that would appear in the JSON
    gold = tp + k["wrong_value"] + k["missed"]
    cats = Counter(p["category"] for p in pairs if p["category"])
    return {"pairs": dict(k), "value_precision": round(tp / predicted, 4) if predicted else None,
            "value_recall": round(tp / gold, 4) if gold else None,
            "value_f1": round(2 * tp / (predicted + gold), 4) if predicted + gold else None,
            "wrong_value_categories": {c: cats.get(c, 0) for c in CATEGORY_ORDER},
            "missed_from_domain_errors": sum(1 for p in pairs if p["kind"] == "missed" and p["domain_error"])}


def summarize_s3(rows_out, s1_row_acc):
    n = len(rows_out)
    stage = Counter(r["failed_stage"] for r in rows_out)
    domain_ok = sum(r["domain_ok"] for r in rows_out)
    field_ok = sum(r["field_set_ok"] for r in rows_out)
    json_ok = sum(r["json_exact"] for r in rows_out)
    on_field_ok = [r for r in rows_out if r["field_set_ok"]]
    value_fail = [r for r in on_field_ok if not r["json_exact"]]
    prim = Counter(r["primary_category"] for r in value_fail)
    groups = Counter(CATEGORY_GROUP[c] for c in prim.elements() if c)
    any_cat = Counter(c for r in value_fail for c in set(r["value_categories"]))
    return {
        "n_rows": n,
        "json_exact": round(json_ok / n, 4),
        "waterfall": {"rows": n, "domain_correct": domain_ok, "field_set_exact": field_ok,
                      "json_exact": json_ok,
                      "json_exact_on_field_set_exact_rows": sum(r["json_exact"] for r in on_field_ok),
                      "rescued_rows (field set wrong, JSON exact: spurious field had no value)":
                          sum(r["rescued"] for r in rows_out)},
        "first_failing_stage": {k: stage.get(k, 0) for k in ("domain", "field_set", "values", "json_exact")},
        "counterfactuals": {
            "actual_json_exact": round(json_ok / n, 4),
            "perfect_extraction (= domain + field set exact)": round(field_ok / n, 4),
            "gold_fields_supplied (scenario 1 row level, searches only)": s1_row_acc,
        },
        "gap_field_set_to_json": {
            "points": round((field_ok - sum(r["json_exact"] for r in on_field_ok)) / n * 100, 2),
            "rows": len(value_fail),
            "by_primary_category": {c: prim.get(c, 0) for c in CATEGORY_ORDER},
            "by_group": dict(groups),
            "rows_with_any_error_of_category": {c: any_cat.get(c, 0) for c in CATEGORY_ORDER},
        },
        "by_language": {lang: round(np.mean([r["json_exact"] for r in rs]), 4)
                        for lang, rs in sorted(_group(rows_out, "language").items(), key=lambda kv: str(kv[0]))},
    }


# ------------------------------------------------------------------ tagger threshold (val_calib)
def tune_tagger():
    rows = load_rows("val_calib", "v2")
    best = None
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        ex = FieldExtractor.load(use_tagger=True)
        ex.tagger.threshold = thr
        acc = summarize_s1(scenario1(ex, rows), rows)["value_accuracy"]
        print(f"[tune] threshold={thr} val_calib scenario-1 accuracy={acc}")
        if best is None or acc > best[1] or (acc == best[1] and abs(thr - 0.5) < abs(best[0] - 0.5)):
            best = (thr, acc)  # ties go to the neutral 0.5
    (ARTIFACTS / "tagger.threshold.json").write_text(json.dumps(
        {"threshold": best[0], "val_calib_scenario1_accuracy": best[1], "tuned_on": "val_calib"}), encoding="utf-8")
    print(f"[tune] chosen threshold {best[0]}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-tagger", action="store_true")
    ap.add_argument("--final-test", action="store_true", help="one-time test evaluation (not used yet)")
    a = ap.parse_args()
    split = "test" if a.final_test else "val_eval"
    protected = [paths.DATA_DIR / f for f in ("train.jsonl", "val.jsonl", "test.jsonl")] + [BUNDLE]
    before = {p.name: sha(p) for p in protected}
    if a.tune_tagger:
        tune_tagger()

    rows = load_rows(split, "v2")
    preds = classify(split)
    variants = {"lexicon": {}, "hybrid": {"use_tagger": True}}
    summary = {"split": split, "development": split == "val_eval", "classifier": str(BUNDLE.relative_to(paths.REPO)),
               "variants": {}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, kw in variants.items():
        t0 = time.perf_counter()
        ex = FieldExtractor.load(**kw)
        s1 = scenario1(ex, rows)
        s1s = summarize_s1(s1, rows)
        pairs, rows_out = scenario2_3(ex, rows, preds)
        s2s = summarize_s2(pairs)
        s3s = summarize_s3(rows_out, s1s["row_all_values_correct"])
        secs = time.perf_counter() - t0
        summary["variants"][name] = {"scenario1": s1s, "scenario2": s2s, "scenario3": s3s,
                                     "seconds": round(secs, 1),
                                     "tagger_threshold": getattr(ex.tagger, "threshold", None)}
        for fname, data in (("s1_records", s1), ("s2_pairs", pairs), ("s3_rows", rows_out)):
            with open(RESULTS / f"{name}_{split}_{fname}.jsonl", "w", encoding="utf-8") as f:
                for x in data:
                    f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")
        print(f"[{name}] S1 value acc={s1s['value_accuracy']} row={s1s['row_all_values_correct']} | "
              f"S2 F1={s2s['value_f1']} | S3 JSON={s3s['json_exact']} field-set={s3s['waterfall']['field_set_exact']}"
              f"/{s3s['n_rows']} ({secs:.0f}s)", flush=True)
    after = {p.name: sha(p) for p in protected}
    assert before == after, "data files or the frozen classifier changed"
    summary["protected_sha256_16"] = after
    (RESULTS / f"summary_{split}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
