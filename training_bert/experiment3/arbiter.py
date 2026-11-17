"""Step 1: a per-field arbiter between the tagger and Frozen Fusion, when they disagree.

    python -m training_bert.experiment3.arbiter [--with-test]

Inputs:
- experiment2/results/predictions_clean_<split>.jsonl: the tagger (3-seed ensemble, cleaned queries, extractor
  fallback)
- experiment2/results/ff_<split>.jsonl: Frozen Fusion on the raw query

When both predict the same domain, every field where they differ is one decision:
    add      Frozen Fusion has the field, the tagger does not   -> take Frozen Fusion's field?
    drop     the tagger has the field, Frozen Fusion does not   -> drop the tagger's field?
    switch   both have it, different values                     -> take Frozen Fusion's value?
A logistic model scores "Frozen Fusion's option is the right one".
- Fitting: val_eval decisions where exactly one option matches gold.
- Threshold: chosen on 5-fold out-of-fold scores (folds by family) to maximise val_eval JSON exact. The final model
  is refitted on all of val_eval.
- Check: val_calib (never used for any choice). Test only with --with-test.
Writes results/arbiter_<split>.jsonl (merged parse per row, same record format) and results/arbiter_report.json.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.extraction.types import values_equal
from src.local_parser import VEHICLE_KEYS, public_payload

from ..common import data as D
from ..common.normalize import SpanNormalizer, dump

HERE = Path(__file__).resolve().parent
E2 = HERE.parent / "experiment2" / "results"
OUT = HERE / "results"
KINDS = ("add", "drop", "switch")
TYPES = ("numeric", "bool", "enum", "text", "date", None)
DOMS = ("נדל״ן", "רכב", "יד_שנייה")
INTERNAL = {v: k for k, v in VEHICLE_KEYS.items()}   # public vehicle key -> internal (v2) key


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def pub(domain, params):
    return public_payload({"category": domain, "params": params})["params"]


def rows_of(split: str):
    """Per row: gold (public), tagger parse (public), Frozen Fusion parse (public), signals."""
    T = jl(E2 / f"predictions_clean_{split}.jsonl")
    F = {r["id"]: r for r in jl(E2 / f"ff_{split}.jsonl")}
    fam = {r["id"]: r["family"] for r in D.load(split)}
    out = []
    for t in T:
        f = F[t["id"]]
        gold = pub(t["domain"], t["gold"]) if t["domain"] else None
        tdom = t["pred_domain"]
        tp = pub(tdom, t["pred"]) if tdom not in (None, "none") and t["pred"] is not None else None
        fdom = (f["pred"] or {}).get("category")
        fp = (f["pred"] or {}).get("params") if f["pred"] else None
        out.append({"id": t["id"], "family": fam[t["id"]], "gold_domain": t["domain"], "gold": gold,
                    "tdom": tdom, "tp": tp, "fdom": fdom, "fp": fp, "raw": t["raw"], "ff_score": f["score"]})
    return out


def decisions(r, norm):
    """(field, kind, tagger option, FF option, features) for every disputed field of a row."""
    if r["tdom"] in (None, "none") or r["fdom"] != r["tdom"]:
        return []
    d, tp, fp = r["tdom"], r["tp"] or {}, r["fp"] or {}
    diff = [k for k in set(tp) | set(fp) if not (k in tp and k in fp and values_equal(tp[k], fp[k]))]
    c = r["raw"].get("conf") or {}
    span_fields = {sp[2] for sp in r["raw"]["spans"]}
    out = []
    for k in diff:
        kind = "add" if k not in tp else "drop" if k not in fp else "switch"
        f = f"{d}/{INTERNAL.get(k, k) if d == 'רכב' else k}"
        ft = norm.ftype(f)
        x = [float(kind == x) for x in KINDS] + [float(ft == x) for x in TYPES] + [float(d == x) for x in DOMS] + [
            r["raw"]["present"].get(f, 0.0), float(f in span_fields), c.get("domain", 1.0), c.get("token_min", 1.0),
            c.get("op_min", 1.0), c.get("agree", 1) / max(1, c.get("members", 1)), r["ff_score"], float(len(diff)),
            float(len(tp)), float(len(fp))]
        out.append((k, kind, tp.get(k), fp.get(k), x))
    return out


def right(option, gold_value):
    return option is None if gold_value is None else option is not None and values_equal(option, gold_value)


def merge(r, decs, probs, th):
    if r["tdom"] in (None, "none"):
        return None
    params = dict(r["tp"] or {})
    for (k, kind, t_opt, f_opt, _), p in zip(decs, probs):
        if p >= th:
            if f_opt is None:
                params.pop(k, None)
            else:
                params[k] = f_opt
    return params


def exact(r, params):
    if r["gold_domain"] is None:
        return r["tdom"] in (None, "none") or not params
    if r["tdom"] != r["gold_domain"] or params is None:
        return False
    return set(params) == set(r["gold"]) and all(values_equal(params[k], v) for k, v in r["gold"].items())


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced").fit((X - mu) / sd, y)
    return lambda Z: m.predict_proba((Z - mu) / sd)[:, 1] if len(Z) else np.zeros(0)


def training_set(rows, norm):
    X, y, fams = [], [], []
    for r in rows:
        g = r["gold"] or {}
        for k, kind, t_opt, f_opt, x in decisions(r, norm):
            tr, fr = right(t_opt, g.get(k)), right(f_opt, g.get(k))
            if tr != fr:
                X.append(x)
                y.append(int(fr))
                fams.append(r["family"])
    return np.array(X), np.array(y), fams


def json_exact(rows, decs, probs, th):
    return float(np.mean([exact(r, merge(r, d, p, th)) for r, d, p in zip(rows, decs, probs)]))


def main(with_test: bool):
    norm = SpanNormalizer.load()
    ev, cal = rows_of("val_eval"), rows_of("val_calib")
    X, y, fams = training_set(ev, norm)
    # out-of-fold scores on val_eval (folds by family) -> threshold
    uf = sorted(set(r["family"] for r in ev))
    random.Random(42).shuffle(uf)
    fold = {f: i % 5 for i, f in enumerate(uf)}
    decs_ev = [decisions(r, norm) for r in ev]
    oof = [np.zeros(len(d)) for d in decs_ev]
    for k in range(5):
        tr = np.array([fold[f] != k for f in fams])
        score = fit(X[tr], y[tr])
        for i, (r, d) in enumerate(zip(ev, decs_ev)):
            if fold[r["family"]] == k and d:
                oof[i] = score(np.array([x for *_, x in d]))
    grid = [round(t, 2) for t in np.arange(0.3, 1.0001, 0.05)] + [1.01]
    curve = {t: json_exact(ev, decs_ev, oof, t) for t in grid}
    th = max(curve, key=lambda t: (curve[t], t))   # ties -> the more conservative threshold
    score = fit(X, y)
    report = {"decisions_val_eval": {"total": int(sum(len(d) for d in decs_ev)), "fit_examples": int(len(y)),
                                     "ff_right_share": float(y.mean())},
              "oof_curve_val_eval": curve, "threshold": th, "splits": {}}
    for split, rows in (("val_eval", ev), ("val_calib", cal)) + ((("test", rows_of("test")),) if with_test else ()):
        decs = [decisions(r, norm) for r in rows]
        probs = [score(np.array([x for *_, x in d])) if d else np.zeros(0) for d in decs]
        base = json_exact(rows, decs, probs, 1.01)
        merged = json_exact(rows, decs, probs, th)
        flips = Counter()
        with open(OUT / f"arbiter_{split}.jsonl", "w", encoding="utf-8") as fh:
            for r, d, p in zip(rows, decs, probs):
                a, b = exact(r, merge(r, d, p, 1.01)), exact(r, merge(r, d, p, th))
                flips["fixed" if b and not a else "broken" if a and not b else "same"] += 1
                fh.write(json.dumps({"id": r["id"], "domain": r["gold_domain"], "gold": r["gold"], "pred_domain": r["tdom"],
                                     "pred": merge(r, d, p, th), "exact": b, "tagger_exact": a, "raw": r["raw"],
                                     "switched": [k for (k, *_), q in zip(d, p) if q >= th]},
                                    ensure_ascii=False) + "\n")
        report["splits"][split] = {"json_exact_tagger": base, "json_exact_arbiter": merged, "rows": dict(flips)}
        print(f"{split}: tagger {base:.4f} -> arbiter {merged:.4f}  {dict(flips)}", flush=True)
    print("threshold", th, "oof curve", {t: round(v, 4) for t, v in curve.items()})
    dump(report, OUT / ("arbiter_report_with_test.json" if with_test else "arbiter_report.json"))


if __name__ == "__main__":
    main("--with-test" in sys.argv[1:])
