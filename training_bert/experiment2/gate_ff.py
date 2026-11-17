"""2A: does Frozen Fusion as a second opinion let the tagger's gate accept more at the same reliability?

    python -m training_bert.experiment2.gate_ff [--clean] [--with-test]

--clean reads the tagger's predictions on runtime-cleaned queries (clean_eval.py) instead of the raw ones;
Frozen Fusion always sees the raw query (an independent second opinion).

Inputs:
- experiment1/runs/e12_gate/predictions_gate_<split>.jsonl: the final tagger (3-seed ensemble + fallback),
  with its confidence signals
- experiment2/results/ff_<split>.jsonl: Frozen Fusion's parse of the same rows (ff_predictions.py)

Gates compared under the experiment-1 protocol (fit on val_eval, threshold on val_calib half A for
target + 0.01 on the one-sided 95% Wilson lower bound, audit on half B, same family split, seed 42):
    tagger      the E12 features (must reproduce E12)
    tagger+ff   + agreement with Frozen Fusion (full JSON, domain, field set) and Frozen Fusion's own gate score
    agree_rule  no model: accept exactly when the tagger and Frozen Fusion produce the same JSON
Test is scored only with --with-test (once, after the comparison on val_calib is final).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.tagger.gate import ff_features, tagger_json  # noqa: F401 (shared with the runtime parser)

from ..common import data as D
from ..common.normalize import dump
from ..gate import features, operating_point, select, wilson_lower

HERE = Path(__file__).resolve().parent
E12 = HERE.parent / "experiment1" / "runs" / "e12_gate"
TARGETS = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97)


CLEAN = False


def load(split):
    src = HERE / "results" / f"predictions_clean_{split}.jsonl" if CLEAN else E12 / f"predictions_gate_{split}.jsonl"
    recs = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()]
    ff = {x["id"]: x for x in map(json.loads, (HERE / "results" / f"ff_{split}.jsonl").read_text(encoding="utf-8").splitlines())}
    base = np.array([features(r) for r in recs])
    extra = np.array([ff_features(r, ff[r["id"]]) for r in recs])
    y = np.array([bool(r["exact"]) for r in recs])
    return recs, base, extra, y


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=4000).fit((X - mu) / sd, y)
    return lambda Z: m.predict_proba((Z - mu) / sd)[:, 1]


def main(with_test: bool):
    _, bf, ef, yf = load("val_eval")
    calib, bc, ec, yc = load("val_calib")
    if with_test:
        _, bt, et, yt = load("test")
    else:   # test is not read at all until the final check
        bt, et, yt = bf[:1] * np.nan, ef[:1] * np.nan, yf[:1]
    fam = {r["id"]: r["family"] for r in D.load("val_calib")}
    fams = sorted({fam[r["id"]] for r in calib})
    random.Random(42).shuffle(fams)
    ia = np.array([fam[r["id"]] in set(fams[: len(fams) // 2]) for r in calib])

    report = {"rows": {"fit": len(yf), "select": int(ia.sum()), "audit": int((~ia).sum()),
                       "test": len(yt) if with_test else None},
              "agreement_rate": {"val_eval": float(ef[:, 0].mean()), "val_calib": float(ec[:, 0].mean()),
                                 "test": float(et[:, 0].mean()) if with_test else None},
              "accuracy_when_agree": {"val_eval": float(yf[ef[:, 0] == 1].mean()), "val_calib": float(yc[ec[:, 0] == 1].mean())},
              "accuracy_when_disagree": {"val_eval": float(yf[ef[:, 0] == 0].mean()), "val_calib": float(yc[ec[:, 0] == 0].mean())},
              "gates": {}}
    for name, (Xf, Xc, Xt) in {"tagger": (bf, bc, bt),
                               "tagger+ff": (np.hstack([bf, ef]), np.hstack([bc, ec]), np.hstack([bt, et]))}.items():
        score = fit(Xf, yf)
        sa, sb = score(Xc[ia]), score(Xc[~ia])
        st = score(Xt) if with_test else None
        curve = {}
        for tgt in TARGETS:
            op = select(sa, yc[ia], tgt + 0.01)
            curve[f"{tgt:.2f}"] = None if op is None else {
                "select": op, "audit": operating_point(sb, yc[~ia], op["threshold"]),
                **({"test": operating_point(st, yt, op["threshold"])} if with_test else {})}
        report["gates"][name] = curve
    # the rule: accept iff the two systems agree (one fixed operating point, no threshold)
    rule = {}
    for split, e, y, mask in (("select", ec, yc, ia), ("audit", ec, yc, ~ia)) + ((("test", et, yt, None),) if with_test else ()):
        acc = e[:, 0] == 1 if mask is None else (e[:, 0] == 1) & mask
        n = len(y) if mask is None else int(mask.sum())
        k, m = int(y[acc].sum()), int(acc.sum())
        rule[split] = {"coverage": m / n, "accuracy": k / m if m else None, "accuracy_lower_95": wilson_lower(k, m)}
    report["agree_rule"] = rule
    name = "gate_ff_report" + ("_clean" if CLEAN else "") + ("_with_test" if with_test else "")
    report["tagger_queries"] = "cleaned" if CLEAN else "raw"
    dump(report, HERE / "results" / f"{name}.json")

    print("agreement", json.dumps(report["agreement_rate"]), "acc|agree", json.dumps(report["accuracy_when_agree"]),
          "acc|disagree", json.dumps(report["accuracy_when_disagree"]))
    for name, curve in report["gates"].items():
        for t, v in curve.items():
            if v is None:
                print(f"{name:10s} {t}: not reachable")
                continue
            s, a = v["select"], v["audit"]
            line = (f"{name:10s} {t}: select cov {s['coverage']:.3f} acc {s['accuracy']:.3f} lb {s['accuracy_lower_95']:.3f}"
                    f" | audit cov {a['coverage']:.3f} acc {a['accuracy']:.3f} lb {a['accuracy_lower_95']:.3f}")
            if "test" in v:
                line += f" | test cov {v['test']['coverage']:.3f} acc {v['test']['accuracy']:.3f}"
            print(line)
    for k, v in rule.items():
        print(f"agree_rule {k}: cov {v['coverage']:.3f} acc {v['accuracy']:.3f} lb {v['accuracy_lower_95']:.3f}")


if __name__ == "__main__":
    CLEAN = "--clean" in sys.argv[1:]
    main("--with-test" in sys.argv[1:])
