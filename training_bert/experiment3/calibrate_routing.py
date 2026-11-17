"""Step 3: calibrate the tagger's gate on the routing batch, with the batch's own protocol.

    python -m training_bert.experiment3.calibrate_routing [--target 0.95] [--with-test]

Protocol: the one the routing batch was built for (README; scripts/calibrate_confidence.py), unchanged.
- Partition: the recorded fit / selection / audit ids (by unit, seed 42) from docs/confidence_audit.json.
- Model: StandardScaler + logistic regression on the fit part.
- Threshold: the largest coverage on the selection part whose one-sided 95% Wilson lower bound, counted over
  accepted units, reaches target + 0.01.
- Audit: the audit part, once, with the threshold unchanged.

Gates compared:
- tagger: the tagger's own signals
- tagger+ff: plus Frozen Fusion's second opinion
- expt2_val_gate: the experiment-2 gate (fitted on val_eval, threshold 0.818), applied as is to the routing audit.
  This measures how far a val-calibrated threshold drifts on test-like data.
--with-test applies the routing-calibrated thresholds to test, once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.calibrate_confidence import operating_point, select_threshold

from ..common.normalize import dump
from ..experiment2.gate_ff import ff_features
from ..gate import features

HERE = Path(__file__).resolve().parent
E2 = HERE.parent / "experiment2" / "results"
OUT = HERE / "results"


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def matrices(recs, ff):
    base = np.array([features(r) for r in recs])
    extra = np.array([ff_features(r, ff[r["id"]]) for r in recs])
    y = np.array([bool(r["exact"]) for r in recs])
    return base, np.hstack([base, extra]), y


def fit(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=4000, random_state=42)).fit(X, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--with-test", action="store_true")
    a = ap.parse_args()
    sel_target = round(a.target + 0.01, 4)

    recs = jl(OUT / "routing_tagger.jsonl")
    ff = {r["id"]: r for r in jl(OUT / "routing_ff.jsonl")}
    parts = {p: [r for r in recs if r["part"] == p] for p in ("fit", "selection", "audit")}
    mats = {p: matrices(rs, ff) for p, rs in parts.items()}
    groups = {p: np.array([r["unit"] for r in rs]) for p, rs in parts.items()}
    elig = {p: np.ones(len(rs), dtype=bool) for p, rs in parts.items()}
    quality = {p: np.array([r.get("quality") or "?" for r in rs]) for p, rs in parts.items()}

    if a.with_test:
        trecs = jl(E2 / "predictions_clean_test.jsonl")
        tff = {r["id"]: r for r in jl(E2 / "ff_test.jsonl")}
        tb, tbf, ty = matrices(trecs, tff)
        tgroups = np.array([r["id"] for r in trecs])

    report = {"target": a.target, "selection_target": sel_target,
              "rows": {p: len(rs) for p, rs in parts.items()},
              "base_accuracy": {p: float(np.mean(mats[p][2])) for p in parts}, "gates": {}}
    for name, col in (("tagger", 0), ("tagger+ff", 1)):
        m = fit(mats["fit"][col], mats["fit"][2])
        ps = m.predict_proba(mats["selection"][col])[:, 1]
        th = select_threshold(ps, mats["selection"][2], elig["selection"], groups["selection"], sel_target)
        g = {"threshold": th}
        if th is not None:
            g["selection"] = operating_point(ps, mats["selection"][2], elig["selection"], groups["selection"], th)
            pa = m.predict_proba(mats["audit"][col])[:, 1]
            g["audit"] = operating_point(pa, mats["audit"][2], elig["audit"], groups["audit"], th)
            g["audit"]["passes_target"] = g["audit"]["accuracy_lower_95"] >= a.target
            g["audit_by_quality"] = {q: operating_point(pa[quality["audit"] == q], mats["audit"][2][quality["audit"] == q],
                                                        elig["audit"][quality["audit"] == q],
                                                        groups["audit"][quality["audit"] == q], th)
                                     for q in sorted(set(quality["audit"]))}
            if a.with_test:
                pt = m.predict_proba(tbf if col else tb)[:, 1]
                g["test"] = operating_point(pt, ty, np.ones(len(ty), dtype=bool), tgroups, th)
        report["gates"][name] = g

    # the experiment-2 val-calibrated gate (tagger+ff, threshold 0.818) on the routing audit, unchanged
    vrecs = jl(E2 / "predictions_clean_val_eval.jsonl")
    vff = {r["id"]: r for r in jl(E2 / "ff_val_eval.jsonl")}
    _, vX, vy = matrices(vrecs, vff)
    e2 = json.loads((E2 / "gate_ff_report_clean_with_test.json").read_text(encoding="utf-8"))
    th2 = e2["gates"]["tagger+ff"]["0.95"]["select"]["threshold"]
    mu, sd = vX.mean(0), vX.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=4000).fit((vX - mu) / sd, vy)   # same fit as experiment2/gate_ff.py
    pa2 = lr.predict_proba((mats["audit"][1] - mu) / sd)[:, 1]
    report["expt2_val_gate_on_routing_audit"] = {"threshold": th2, **operating_point(
        pa2, mats["audit"][2], elig["audit"], groups["audit"], th2)}

    name = "calibration_report" + ("_with_test" if a.with_test else "")
    dump(report, OUT / f"{name}.json")
    print("base accuracy", json.dumps({k: round(v, 4) for k, v in report["base_accuracy"].items()}))
    for gname, g in report["gates"].items():
        if g["threshold"] is None:
            print(f"{gname}: no threshold reaches {sel_target} on the selection part")
            continue
        s, au = g["selection"], g["audit"]
        line = (f"{gname:10s} th {g['threshold']:.3f} | selection cov {s['coverage']:.3f} acc {s['accuracy']:.3f} "
                f"lb {s['accuracy_lower_95']:.3f} | audit cov {au['coverage']:.3f} acc {au['accuracy']:.3f} "
                f"lb {au['accuracy_lower_95']:.3f} pass {au['passes_target']}")
        if "test" in g:
            line += f" | test cov {g['test']['coverage']:.3f} acc {g['test']['accuracy']:.3f}"
        print(line)
        print("   audit by quality", {q: (round(v["coverage"], 3), round(v["accuracy"] or 0, 3), v["accepted"])
                                      for q, v in g["audit_by_quality"].items()})
    e = report["expt2_val_gate_on_routing_audit"]
    print(f"experiment-2 val gate (th {th2:.3f}) on routing audit: cov {e['coverage']:.3f} acc {e['accuracy']:.3f} "
          f"lb {e['accuracy_lower_95']:.3f}")


if __name__ == "__main__":
    main()
