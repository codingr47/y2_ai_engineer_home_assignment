"""Exp6: does typo augmentation of train improve the fusion classifier's field recall on fresh phrasings?

Same model family and frozen config as Exp3 v2 (E5 + TF-IDF early fusion, LR heads, C=16, w=1). The only change is
training data: each train row gets k noisy copies (letter deletion / duplication / swap / confusable substitution,
glued or split words; digits untouched) with the SAME labels. Nothing else moves:

    train        fits the heads and the TF-IDF vocabulary (original + noisy copies)
    val_calib    calibration and field thresholds (unchanged rows)
    val_eval     selection between k values (DEVELOPMENT; split1 distribution)
    routing dev  routing batch fit + selection partitions (DEVELOPMENT; fresh distribution). The routing audit
                 partition stays sealed; test is not scored.

Metric: end-to-end JSON exact with the frozen extractor (search rows), plus classifier field-set exact.

python training_frozen_fusion/experiment_6_typo_augment/run.py            # k = 0 (rebuild of Exp3 v2), 1, 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training_frozen_fusion"))
sys.path.insert(0, str(REPO / "training_frozen_fusion" / "experiment_5_routing_errors"))

from shared import paths  # noqa: E402,F401  (experiment_1 on sys.path)
from shared.dataset import load_rows  # noqa: E402
from shared.features import make_featurizer  # noqa: E402
from shared.runner import Bundle, base_split, relabel  # noqa: E402
from shared.slots import slot_vocab  # noqa: E402
from shared.taxonomy_rules import Resolver  # noqa: E402
from train import Heads, Split, load_vocab  # noqa: E402

from analyze import dev_rows  # noqa: E402
from scripts.export_classifier import export  # noqa: E402
from src.classifier import FusionClassifier  # noqa: E402
from src.extraction.extractor import FieldExtractor  # noqa: E402
from src.extraction.types import values_equal  # noqa: E402

OUT = HERE / "results"
ART = HERE / "artifacts"
C, W = 16, 1.0
DOMAINS = ("נדל״ן", "רכב", "יד_שנייה")

HEBREW = set("אבגדהוזחטיכלמנסעפצקרשתךםןףץ")
CONFUSABLE = {"ט": "ת", "ת": "ט", "כ": "ק", "ק": "כ", "ס": "ש", "ש": "ס", "א": "ע", "ע": "א", "ה": "א",
              "ב": "ו", "ו": "ב", "ח": "כ", "ם": "מ", "מ": "ם", "ן": "נ", "נ": "ן", "ך": "כ", "ף": "פ", "ץ": "צ"}


def noisy(query: str, rng: random.Random) -> str:
    """1-2 typo operations on Hebrew words of 3+ letters. Digits, Latin and punctuation are never touched, so every
    number and model code in the query keeps its meaning and the labels stay valid."""
    words = query.split(" ")
    for _ in range(rng.choice((1, 1, 2))):
        idx = [i for i, w in enumerate(words) if sum(ch in HEBREW for ch in w) >= 3]
        if not idx:
            break
        i = rng.choice(idx)
        w = words[i]
        pos = [j for j, ch in enumerate(w) if ch in HEBREW]
        j = rng.choice(pos)
        op = rng.choice(("delete", "double", "swap", "confuse", "confuse", "glue", "split"))
        if op == "delete":
            w = w[:j] + w[j + 1:]
        elif op == "double":
            w = w[:j] + w[j] + w[j:]
        elif op == "swap" and j + 1 < len(w) and w[j + 1] in HEBREW:
            w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        elif op == "confuse" and w[j] in CONFUSABLE:
            w = w[:j] + CONFUSABLE[w[j]] + w[j + 1:]
        elif op == "glue" and i + 1 < len(words):
            words[i:i + 2] = [w + words[i + 1]]
            continue
        elif op == "split" and 0 < j < len(w) - 1:
            w = w[:j] + " " + w[j:]
        words[i] = w
    out = " ".join(words)
    return out if out.strip() else query


def augmented_train(k: int, seed: int = 0) -> Split:
    from embed import encode
    train = base_split("train")
    if k == 0:
        return train
    rng = random.Random(seed)
    texts = [noisy(m["query"], rng) for _ in range(k) for m in train.meta]
    cache = ART / f"emb_train_aug_k{k}_seed{seed}.npy"
    if cache.exists():
        E = np.load(cache)
    else:
        E = encode(texts)
        ART.mkdir(parents=True, exist_ok=True)
        np.save(cache, E)
    (ART / f"train_aug_k{k}_seed{seed}.sample.json").write_text(
        json.dumps([[m["query"], t] for m, t in list(zip(train.meta, texts))[:40]], ensure_ascii=False, indent=1),
        encoding="utf-8")
    meta = [{**m, "query": t} for m, t in zip([m for _ in range(k) for m in train.meta], texts)]
    labels = [r for _ in range(k) for r in train.labels]
    return Split("train", np.vstack([train.X, E]), train.labels + labels, train.meta + meta)


def fit_bundle(k: int) -> Bundle:
    vocab77 = load_vocab()
    sub = augmented_train(k)
    feat = make_featurizer("e5+tfidf", W).fit(sub)
    tr = relabel(sub, feat.transform(sub), "v2", vocab77)
    calib = base_split("val_calib")
    ca = relabel(calib, feat.transform(calib), "v2", vocab77)
    from shared.dataset import field_vocab
    heads = Heads(vocab=list(field_vocab("v2")), C=C).fit(tr).calibrate(ca)
    resolver = Resolver.from_train(base_split("train").fields, slot_vocab(vocab77))
    return Bundle(f"exp6-k{k}", "e5+tfidf", "v2", C, W, feat, heads, resolver, vocab77)


class CachedEncoder:
    """Serves precomputed E5 vectors to FusionClassifier.features (same prefix and normalisation)."""
    def __init__(self, table):
        self.table = table

    def encode(self, texts, **_):
        return np.stack([self.table[t] for t in texts])


def eval_sets():
    sets = {"val_eval": [(r["query"], r["domain"], (r.get("expected") or {}).get("params") or {}, "standard")
                         for r in load_rows("val_eval")]}
    dev, _ = dev_rows()
    sets["routing_dev"] = [(r.query, r.domain, r.expected["params"], r.language) for r, _ in dev]
    return sets


def embeddings(sets):
    from embed import encode
    path = ART / "emb_eval_sets.npz"
    texts = sorted({"query: " + q for rows in sets.values() for q, *_ in rows})
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if list(z["texts"]) == texts:
            return dict(zip(texts, z["E"]))
    from embed import get_encoder
    E = get_encoder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False,
                             convert_to_numpy=True).astype(np.float32)
    ART.mkdir(parents=True, exist_ok=True)
    np.savez(path, texts=np.array(texts, dtype=object), E=E)
    return dict(zip(texts, E))


def evaluate(artifact, table, rows, extractor):
    clf = FusionClassifier(artifact, CachedEncoder(table))
    queries = [q for q, *_ in rows]
    pd, pf, sel = clf.predict_X(clf.features(queries))
    vocab, classes = artifact["vocab"], artifact["classes"]
    res = {"search": 0, "json_exact": 0, "field_set_exact": 0, "by_language": {}}
    for i, (q, dom, gold, lang) in enumerate(rows):
        if dom is None:
            continue
        pdom = str(classes[pd[i].argmax()])
        fields = [f for f, s in zip(vocab, sel[i]) if s]
        fs_ok = pdom == dom and {f.split("/", 1)[1] for f in fields} == set(gold)
        params = None
        if pdom in DOMAINS:
            params = (extractor.to_json(pdom, extractor.extract_all(q, pdom, fields)) or {}).get("params")
        ok = pdom == dom and params is not None and set(params) == set(gold) and all(
            values_equal(params[k], v) for k, v in gold.items())
        res["search"] += 1
        res["json_exact"] += ok
        res["field_set_exact"] += fs_ok
        lg = res["by_language"].setdefault(lang, [0, 0])
        lg[0] += 1
        lg[1] += ok
    n = res["search"]
    return {"search_rows": n, "json_exact": round(res["json_exact"] / n, 4),
            "field_set_exact": round(res["field_set_exact"] / n, 4),
            "json_exact_by_language": {k: round(v[1] / v[0], 4) for k, v in sorted(res["by_language"].items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    sets = eval_sets()
    table = embeddings(sets)
    extractor = FieldExtractor.load(use_tagger=True)
    OUT.mkdir(exist_ok=True)
    results = {}
    frozen = __import__("joblib").load(REPO / "src/artifacts/classifier.joblib")
    results["shipped_exp3_v2"] = {n: evaluate(frozen, table, rows, extractor) for n, rows in sets.items()}
    print("shipped", json.dumps(results["shipped_exp3_v2"], ensure_ascii=False))
    for k in args.k:
        t = time.time()
        bundle = fit_bundle(k)
        artifact = export(bundle, ART / f"classifier_k{k}.joblib")
        results[f"k{k}"] = {n: evaluate(artifact, table, rows, extractor) for n, rows in sets.items()}
        results[f"k{k}"]["fit_seconds"] = round(time.time() - t, 1)
        print(f"k={k}", json.dumps(results[f"k{k}"], ensure_ascii=False))
    (OUT / "summary_dev.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
