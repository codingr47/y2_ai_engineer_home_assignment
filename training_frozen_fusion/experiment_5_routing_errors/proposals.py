"""Extractor-proposed fields: can the extractor recover fields the frozen classifier rejected?

For every in-domain field the classifier did NOT select, run the field-local extractor and record what it finds,
against gold. Cached per data set, then policies are compared:

    val_calib     -> fits the per-field proposal policy (a threshold: its data role)
    val_eval      -> development evaluation (split1 distribution)
    routing dev   -> development evaluation on the fresh distribution (fit + selection; audit sealed)

python training_frozen_fusion/experiment_5_routing_errors/proposals.py            # cache (slow, ~3 min) + evaluate
python training_frozen_fusion/experiment_5_routing_errors/proposals.py --cached   # evaluate only
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from analyze import dev_rows  # noqa: E402
from src.extraction.types import values_equal  # noqa: E402

CACHE = HERE / "results" / "proposal_cache.pkl"
DOMAINS = ("נדל״ן", "רכב", "יד_שנייה")


def datasets():
    from training_frozen_fusion.shared.dataset import load_rows
    out = {}
    for split in ("val_calib", "val_eval"):
        out[split] = [{"id": r["id"], "query": r["query"], "domain": r["domain"],
                       "gold": (r.get("expected") or {}).get("params") or {}} for r in load_rows(split)]
    dev, _ = dev_rows()
    out["routing_dev"] = [{"id": r.id, "query": r.query, "domain": r.domain, "gold": r.expected["params"],
                           "language": r.language} for r, _ in dev]
    return out


def build_cache():
    from src.classifier import FusionClassifier
    from src.extraction.extractor import FieldExtractor
    from src.settings import Settings
    s = Settings.from_env()
    clf = FusionClassifier.load(s.classifier_artifact_path, s.e5_model_path)
    ex = FieldExtractor.load(use_tagger=s.extractor_use_tagger)
    cache = {}
    for name, rows in datasets().items():
        recs = []
        for r in rows:
            c = clf.classify(r["query"])
            rec = {**r, "pred_domain": c.domain, "selected": list(c.fields), "candidates": {}}
            if c.domain in DOMAINS:
                ctx = ex._context(r["query"], c.domain)
                for f, fi in ex.model.fields.items():
                    if not f.startswith(c.domain + "/") or f in c.fields:
                        continue
                    e = ex.extract(r["query"], f, ctx)
                    if e.found:
                        prob = float(c.field_probs[list(clf.artifact["vocab"]).index(f)]) \
                            if f in clf.artifact["vocab"] else 0.0
                        rec["candidates"][f] = {"value": e.value, "source": e.source, "score": e.score,
                                                "type": fi.type, "prob": prob, "text": e.text}
            recs.append(rec)
        cache[name] = recs
    CACHE.parent.mkdir(exist_ok=True)
    pickle.dump(cache, open(CACHE, "wb"))
    return cache


def proposal_outcome(rec, f, cand):
    name = f.split("/", 1)[1]
    if rec["domain"] != rec["pred_domain"] or name not in rec["gold"]:
        return "false"
    return "right" if values_equal(cand["value"], rec["gold"][name]) else "wrong"


def tally(recs, key):
    t = collections.defaultdict(collections.Counter)
    for rec in recs:
        for f, cand in rec["candidates"].items():
            t[key(f, cand)][proposal_outcome(rec, f, cand)] += 1
    return t


def fit_policy(recs, min_n=3, min_precision=.8, sources=None):
    """Per (field, source): allow proposals when val_calib precision (right / all proposals) >= min_precision."""
    t = tally(recs, lambda f, c: (f, c["source"]))
    allow = {}
    for k, c in t.items():
        n = sum(c.values())
        if sources is not None and k[1] not in sources:
            continue
        if n >= min_n and c["right"] / n >= min_precision:
            allow[k] = {"n": n, **c}
    return allow


def evaluate(recs, extractor, allow, min_prob=0.0):
    """End-to-end JSON exact with proposals added to the selected fields (re-running extract_all for conflicts)."""
    exact = base = 0
    changed = collections.Counter()
    for rec in recs:
        dom = rec["pred_domain"]
        fields = list(rec["selected"])
        add = [f for f, c in rec["candidates"].items() if (f, c["source"]) in allow and c["prob"] >= min_prob]

        def parse(fs):
            if dom not in DOMAINS:
                return None
            out = extractor.extract_all(rec["query"], dom, fs)
            return (extractor.to_json(dom, out) or {}).get("params")

        def ok(params):
            if rec["domain"] is None:
                return False
            return dom == rec["domain"] and params is not None and set(params) == set(rec["gold"]) and all(
                values_equal(params[k], v) for k, v in rec["gold"].items())

        if "_base" not in rec:
            rec["_base"] = parse(fields)
        p0 = rec["_base"]
        b = ok(p0)
        a = ok(parse(fields + add)) if add else b
        base += b
        exact += a
        changed[(b, a)] += 1
    n = sum(1 for r in recs if r["domain"])
    return {"search_rows": n, "base": round(base / n, 4), "with_proposals": round(exact / n, 4),
            "fixed": changed[(False, True)], "broken": changed[(True, False)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    args = ap.parse_args()
    cache = pickle.load(open(CACHE, "rb")) if args.cached else build_cache()
    from src.extraction.extractor import FieldExtractor
    ex = FieldExtractor.load(use_tagger=True)
    for name, recs in cache.items():
        t = tally(recs, lambda f, c: c["source"])
        print(name, {k: dict(v) for k, v in t.items()})
    report = {}
    for label, kw in {"p80": dict(min_precision=.8), "p90": dict(min_precision=.9),
                      "p80_no_tagger_fuzzy": dict(min_precision=.8, sources={"lexicon", "cue", "date", "bool", "unit",
                                                                                "numeric", "taxonomy"})}.items():
        allow = fit_policy(cache["val_calib"], **kw)
        report[label] = {"allowed": len(allow),
                         **{n: evaluate(cache[n], ex, allow) for n in ("val_eval", "routing_dev")}}
        print(label, json.dumps(report[label], ensure_ascii=False))
    (HERE / "results" / "proposals_dev.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
