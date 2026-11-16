"""Exp7: Exp3 v2 retrained on train + the extra training batch (training_frozen_fusion/data/extra_train).

Same model family and frozen config (E5 + TF-IDF early fusion, LR heads, C=16, w=1); C is re-checked on val_eval.
The extra rows get v2 labels at load time (files untouched) and a family-level subsample gives learning-curve
points (0, 50%, 100% of the extra batch).

    train + extra   fit heads and TF-IDF
    val_calib       calibration and thresholds
    val_eval        selection (DEVELOPMENT)
    routing dev     development, fresh distribution (routing audit sealed; test not scored)

python training_frozen_fusion/experiment_7_extra_train/run.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "training_frozen_fusion" / "experiment_6_typo_augment"))

import run as exp6  # noqa: E402  (sys.path setup, evaluation helpers)
from shared.dataset import field_labels, field_vocab  # noqa: E402
from shared.features import make_featurizer  # noqa: E402
from shared.normalize import normalize_row  # noqa: E402
from shared.runner import Bundle, base_split, relabel  # noqa: E402
from shared.slots import slot_vocab  # noqa: E402
from shared.taxonomy_rules import Resolver  # noqa: E402
from train import Heads, Split, load_vocab  # noqa: E402
from run_experiment import family_subsample  # noqa: E402

EXTRA = REPO / "training_frozen_fusion/data/extra_train/train_extra.jsonl"
ART, OUT = HERE / "artifacts", HERE / "results"


def extra_split() -> Split:
    """The extra batch as an experiment_1-style Split with v2 labels and cached E5 vectors."""
    from embed import encode
    rows = [normalize_row(json.loads(line)) for line in EXTRA.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [{"id": r["id"], "domain_label": r["domain"] or "none", "field_labels": field_labels(r)} for r in rows]
    meta = [{"id": r["id"], "query": r["query"], "family": r["unit"], "quality": r["quality"],
             "language": "+".join(r["metadata"]["language"]), "complexity": "+".join(r["metadata"]["complexity"])}
            for r in rows]
    cache = ART / "emb_extra.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        X = encode([m["query"] for m in meta])
        ART.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
    return Split("extra", X, labels, meta)


def combined(extra: Split, size) -> Split:
    base = base_split("train")
    base = relabel(base, base.X, "v2", load_vocab())  # v2 labels for the original train rows
    sub = extra.subset(family_subsample(extra, size, 0)) if size else extra.subset(np.arange(0))
    return Split("train", np.vstack([base.X, sub.X]), base.labels + sub.labels, base.meta + sub.meta)


def fit_bundle(train: Split, C: float) -> Bundle:
    vocab77 = load_vocab()
    feat = make_featurizer("e5+tfidf", exp6.W).fit(train)
    tr = Split("train", feat.transform(train), train.labels, train.meta)
    calib = base_split("val_calib")
    ca = relabel(calib, feat.transform(calib), "v2", vocab77)
    heads = Heads(vocab=list(field_vocab("v2")), C=C).fit(tr).calibrate(ca)
    resolver = Resolver.from_train(base_split("train").fields, slot_vocab(vocab77))
    return Bundle("exp7", "e5+tfidf", "v2", C, exp6.W, feat, heads, resolver, vocab77)


def main():
    sets = exp6.eval_sets()
    table = exp6.embeddings(sets)
    extractor = exp6.FieldExtractor.load(use_tagger=True)
    extra = extra_split()
    OUT.mkdir(exist_ok=True)
    results = {}
    import joblib
    shipped = joblib.load(REPO / "src/artifacts/classifier.joblib")
    results["shipped"] = {n: exp6.evaluate(shipped, table, rows, extractor) for n, rows in sets.items()}
    print("shipped", json.dumps(results["shipped"], ensure_ascii=False), flush=True)
    for label, size, C in (("extra50_C16", len(extra.labels) // 2, 16), ("extra100_C16", None, 16),
                           ("extra100_C64", None, 64)):
        t = time.time()
        train = combined(extra, size if size else len(extra.labels))
        bundle = fit_bundle(train, C)
        artifact = exp6.export(bundle, ART / f"classifier_{label}.joblib")
        results[label] = {n: exp6.evaluate(artifact, table, rows, extractor) for n, rows in sets.items()}
        results[label].update(train_rows=len(train.labels), fit_seconds=round(time.time() - t, 1))
        print(label, json.dumps(results[label], ensure_ascii=False), flush=True)
    (OUT / "summary_dev.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
