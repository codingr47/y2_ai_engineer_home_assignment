"""Confidence gate for the Fine-Tuned Tagger: which parses to answer locally, which to send to the LLM.

    python -m training_bert.gate training_bert/experiment1/runs/e12_gate --target 0.95

Reads <dir>/predictions_gate_<split>.jsonl (final_test --ensemble ... --tag=gate, which also writes the
model's confidence signals) for val_eval, val_calib and test.

Data roles (nothing here touches the model):
    val_eval        fit the gate (logistic regression: features -> "this parse is exactly right")
    val_calib half A  choose the threshold: the largest coverage whose one-sided 95% Wilson lower bound on accuracy
                    reaches target + 0.01 (the same margin as the Frozen Fusion gate)
    val_calib half B  audit that threshold once, unchanged
    test            reported once
Halves are split by family (seed 42), so no family is in both. Writes <dir>/gate_report.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .common import data as D
from .common.normalize import dump

DOMS = ("none", "נדל״ן", "רכב", "יד_שנייה")


def wilson_lower(correct: int, n: int) -> float:
    """One-sided 95% Wilson lower bound (same formula as scripts/calibrate_confidence.py)."""
    if n == 0:
        return 0.0
    z = 1.6448536269514722
    p = correct / n
    return float((p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n))


def features(rec: dict) -> list[float]:
    raw = rec["raw"]
    c = raw.get("conf") or {}
    pres = raw.get("present") or {}
    pred = rec.get("pred") or {}
    d = raw["domain"]
    pred_fields = {f"{d}/{k}" for k in pred}
    span_fields = {sp[2] for sp in raw["spans"]}
    p_in = [pres.get(f, 0.0) for f in pred_fields]
    p_out = [p for f, p in pres.items() if f not in pred_fields]
    return [c.get("domain", 1.0), c.get("token_min", 1.0), c.get("op_min", 1.0),
            c.get("agree", 1) / max(1, c.get("members", 1)), float(c.get("agree", 1) == c.get("members", 1)),
            min(p_in) if p_in else 1.0, max(p_out) if p_out else 0.0,
            len(pred), len(raw["spans"]), float(bool(pred_fields - span_fields - set(raw.get("implied") or {}))),
            np.log1p(len(rec["query"])), *[float(d == x) for x in DOMS]]


def load(dir_: Path, split: str):
    recs = [json.loads(l) for l in (dir_ / f"predictions_gate_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    return recs, np.array([features(r) for r in recs]), np.array([bool(r["exact"]) for r in recs])


def operating_point(scores, y, th):
    acc = scores >= th
    k, n = int(y[acc].sum()), int(acc.sum())
    return {"threshold": float(th), "coverage": n / len(y), "accepted": n, "correct": k,
            "accuracy": k / n if n else None, "accuracy_lower_95": wilson_lower(k, n) if n else None}


def select(scores, y, need: float):
    """Lowest threshold (= most coverage) whose Wilson lower bound reaches `need`; None if none does."""
    best = None
    for th in np.unique(scores)[::-1]:
        op = operating_point(scores, y, th)
        if op["accepted"] >= 30 and op["accuracy_lower_95"] >= need:
            best = op
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--monthly-searches", type=float, default=33_000_000)
    ap.add_argument("--llm-cost-per-1k", type=float, default=1300 / 13_200)   # user's figure: 13.2M -> $1,300
    a = ap.parse_args()

    _, Xf, yf = load(a.dir, "val_eval")
    calib, Xc, yc = load(a.dir, "val_calib")
    test, Xt, yt = load(a.dir, "test")
    fam = {r["id"]: r["family"] for r in D.load("val_calib")}
    fams = sorted({fam[r["id"]] for r in calib})
    random.Random(42).shuffle(fams)
    half_a = set(fams[: len(fams) // 2])
    ia = np.array([fam[r["id"]] in half_a for r in calib])

    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    gate = LogisticRegression(C=1.0, max_iter=2000).fit((Xf - mu) / sd, yf)
    score = lambda X: gate.predict_proba((X - mu) / sd)[:, 1]
    sa, sb, st = score(Xc[ia]), score(Xc[~ia]), score(Xt)

    report = {"target": a.target, "selection_needs_lower_bound": a.target + 0.01,
              "rows": {"fit_val_eval": len(yf), "select_val_calib_A": int(ia.sum()), "audit_val_calib_B": int((~ia).sum()),
                       "test": len(yt)},
              "base_accuracy": {"val_eval": float(yf.mean()), "val_calib": float(yc.mean()), "test": float(yt.mean())},
              "curve": {}}
    for tgt in (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98):
        op = select(sa, yc[ia], tgt + 0.01)
        if op is None:
            report["curve"][f"{tgt:.2f}"] = None
            continue
        report["curve"][f"{tgt:.2f}"] = {"select": op, "audit": operating_point(sb, yc[~ia], op["threshold"]),
                                         "test": operating_point(st, yt, op["threshold"])}
    chosen = report["curve"].get(f"{a.target:.2f}")
    report["chosen"] = chosen
    if chosen:
        cov = chosen["test"]["coverage"]
        llm = a.monthly_searches * (1 - cov)
        report["cost_estimate"] = {"monthly_searches": a.monthly_searches, "local_share_test": cov,
                                   "llm_calls_per_month": llm, "llm_usd_per_month": llm / 1000 * a.llm_cost_per_1k,
                                   "llm_only_usd_per_month": a.monthly_searches / 1000 * a.llm_cost_per_1k}
    dump(report, a.dir / "gate_report.json")
    print(json.dumps(report["base_accuracy"]))
    for t, v in report["curve"].items():
        if v is None:
            print(f"target {t}: not reachable on the selection half")
            continue
        s, au, te = v["select"], v["audit"], v["test"]
        print(f"target {t}: th {s['threshold']:.3f} | select cov {s['coverage']:.3f} acc {s['accuracy']:.3f} "
              f"lb {s['accuracy_lower_95']:.3f} | audit cov {au['coverage']:.3f} acc {au['accuracy']:.3f} "
              f"lb {au['accuracy_lower_95']:.3f} | test cov {te['coverage']:.3f} acc {te['accuracy']:.3f}")
    if report.get("cost_estimate"):
        print(json.dumps(report["cost_estimate"]))


if __name__ == "__main__":
    main()
