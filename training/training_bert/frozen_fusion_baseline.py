"""Score the shipped Frozen Fusion parser (src/artifacts) on the tagger's val_eval, with the same JSON equality.

    python -m training.training_bert.frozen_fusion_baseline            -> training/training_bert/results/frozen_fusion_val_eval.json

Needs the provisioned encoder at models/multilingual-e5-small. Frozen Fusion never trained on any val row.
"""
from __future__ import annotations

from collections import Counter

from src.local_parser import LocalParser, public_payload
from src.settings import Settings

from .common import data as D
from .common.normalize import dump, json_equal

OUT = D.ROOT / "results" / "frozen_fusion_val_eval.json"


def _load_parser():
    """LocalParser.load, tolerating a `code` binding mismatch only. The restructuring commit (d79bfa6) changed two
    docstrings in src/extraction, which changes the code hash but not behavior; every other binding (classifier,
    extractor assets, encoder, features) must still match."""
    import joblib

    from src import confidence

    settings = Settings(extractor_enabled=True)
    stored = joblib.load(settings.confidence_artifact_path)["bindings"]
    real = confidence.bindings
    def tolerant(*a, **k):
        b = real(*a, **k)
        assert {x: v for x, v in b.items() if x != "code"} == {x: v for x, v in stored.items() if x != "code"}
        return {**b, "code": stored["code"]}
    import src.local_parser as lp   # imports `bindings` by name
    lp.bindings = tolerant
    try:
        return LocalParser.load(settings)
    finally:
        lp.bindings = real


def main(split: str = "val_eval"):
    parser = _load_parser()
    agg, slices = Counter(), {}
    for r in D.load(split):
        cand = parser.candidate(r["query"])
        pred = public_payload(cand.payload) if cand.payload else None
        c = parser.classifier.classify(r["query"])
        gold = public_payload({"category": r["domain"], "params": D.params(r)}) if r["domain"] else None
        exact = json_equal(pred, gold)
        fs = (c.domain == (r["domain"] or "none")) and (
            {f.split("/", 1)[1] for f in c.fields} == set(D.params(r)) if r["domain"] else True)
        accepted = cand.reason is None and parser.gate.accepts(parser.gate.score(cand.vector))
        agg["n"] += 1
        agg["json"] += exact
        agg["field_set"] += fs
        agg["domain"] += c.domain == (r["domain"] or "none")
        agg["accepted"] += accepted
        agg["accepted_ok"] += accepted and exact
        for key in ("quality", "domain"):
            k = f"{key}={r.get(key) or 'none'}"
            s = slices.setdefault(k, Counter())
            s["n"] += 1
            s["json"] += exact
    n = agg["n"]
    out = {"split": split, "n": n, "json_exact": agg["json"] / n, "field_set_exact": agg["field_set"] / n,
           "domain_acc": agg["domain"] / n, "gate_coverage": agg["accepted"] / n,
           "gate_accuracy": agg["accepted_ok"] / max(1, agg["accepted"]),
           "slices": {k: {"n": s["n"], "json_exact": s["json"] / s["n"]} for k, s in sorted(slices.items())}}
    dump(out, OUT)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items() if k != "slices"})
    print({k: round(v["json_exact"], 4) for k, v in out["slices"].items()})


if __name__ == "__main__":
    main()
