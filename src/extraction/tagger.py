"""Learned fallback span finder: one token classifier per field, trained on the gold spans of TRAIN.

Used only when the deterministic matchers find nothing for a field. Each token gets hashed features (folded
word, prefix-stripped stem, character 3-grams, neighbouring stems, digit flag); a per-field logistic regression scores "inside this field's span". The best contiguous run above the
threshold becomes the span, and the same deterministic normalizers turn it into a value. The threshold is tuned
on val_calib (training_frozen_fusion/experiment_4_extraction/evaluate_extraction.py --tune-tagger) and stored next to the model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .match import date_matches, fuzzy_matches, lexicon_matches
from .numbers import parse_numeric
from .text import fold, negated_before, prefix_variants, tokens
from .types import Extraction

MIN_SPANS = 15
N_FEATURES = 2 ** 16
_STRIP_CUES = re.compile(r"^(?:ו?ב?(?:רחוב|רח'|שכונת|שכונה|מותג|של|מבית))\s+")


def _stem(t: str) -> str:
    core = prefix_variants(t)[-1]
    return fold(core) if len(core) > 2 else fold(t)


def token_features(qtok, i: int) -> list[str]:
    t = qtok[i].text
    s = _stem(t)
    feats = [f"w={fold(t)}", f"s={s}", f"d={int(any(c.isdigit() for c in t))}",
             f"p={t[0] if t and t[0] in 'ובלמהכש' else '-'}"]
    padded = f"<{s}>"
    feats += [f"g={padded[k:k + 3]}" for k in range(max(1, len(padded) - 2))]
    for off in (-2, -1, 1, 2):
        j = i + off
        feats.append(f"n{off}={_stem(qtok[j].text) if 0 <= j < len(qtok) else '#'}")
    return feats


class SpanTagger:
    def __init__(self, models: dict, threshold: float = 0.5):
        self.models = models          # field -> fitted LogisticRegression
        self.threshold = threshold
        self._hasher = None

    def _hash(self, feats_list):
        if self._hasher is None:
            from sklearn.feature_extraction import FeatureHasher
            self._hasher = FeatureHasher(n_features=N_FEATURES, input_type="string", alternate_sign=False)
        return self._hasher.transform(feats_list)

    @classmethod
    def train(cls, rows: list[dict], model) -> "SpanTagger":
        from sklearn.linear_model import LogisticRegression

        by_domain: dict[str, list] = {}
        for r in rows:
            if r.get("domain"):
                by_domain.setdefault(r["domain"], []).append(r)
        tagger = cls({})
        for f, fi in model.fields.items():
            dom, name = f.split("/", 1)
            X, y, n_spans = [], [], 0
            for r in by_domain.get(dom, []):
                qtok = tokens(r["query"])
                spans = (r.get("spans") or {}).get(name, [])
                n_spans += len(spans)
                for i, t in enumerate(qtok):
                    X.append(token_features(qtok, i))
                    y.append(int(any(t.start < s["end"] and t.end > s["start"] for s in spans)))
            if n_spans < MIN_SPANS or sum(y) == 0:
                continue
            clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
            clf.fit(tagger._hash(X), np.array(y))
            tagger.models[f] = clf
        return tagger

    def save(self, path: Path):
        import joblib

        # compressed: ~76% of the hashed weights are zero, so the file shrinks ~4x (36 MB -> ~9 MB), weights unchanged
        joblib.dump({"models": self.models, "threshold": self.threshold}, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "SpanTagger":
        import joblib

        d = joblib.load(path)
        thr_file = Path(path).with_suffix(".threshold.json")
        thr = json.loads(thr_file.read_text())["threshold"] if thr_file.exists() else d["threshold"]
        return cls(d["models"], thr)

    # ------------------------------------------------------------------ inference
    def best_span(self, query: str, field: str, ctx: dict):
        clf = self.models.get(field)
        qtok = ctx["tokens"]
        if clf is None or not qtok:
            return None
        p = clf.predict_proba(self._hash([token_features(qtok, i) for i in range(len(qtok))]))[:, 1]
        best, cur, best_run, start = 0.0, 0.0, None, 0
        for i, pi in enumerate(p):  # max-sum contiguous run of (p - threshold)
            gain = pi - self.threshold
            if cur <= 0:
                cur, start = gain, i
            else:
                cur += gain
            if cur > best:
                best, best_run = cur, (start, i)
        if best_run is None:
            return None
        a, b = best_run
        return qtok[a].start, qtok[b].end, float(p[a:b + 1].mean()), a

    def extract(self, query: str, field: str, fi, model, ctx: dict) -> Extraction | None:
        hit = self.best_span(query, field, ctx)
        if hit is None:
            return None
        s, e, conf, a = hit
        text = query[s:e]
        value, op = None, None
        if fi.type == "numeric":
            op, value = parse_numeric(text, field)
        elif fi.type == "bool":
            value = not negated_before(ctx["tokens"], a, 2)
        elif fi.type == "date":
            d = date_matches(text)
            value = d[0].value if d else None
        elif fi.type == "text":
            value = _STRIP_CUES.sub("", text).strip(" ,.!?\"'") or None
            if value and value[0] in "ב" and len(value) > 3:
                value = value[1:]
        else:  # enum: map the tagged text onto a known surface (exact, then one edit away)
            sub = tokens(text)
            ms = lexicon_matches(fi, text, sub) or fuzzy_matches(fi, text, sub)
            value = ms[0].value if ms else None
        if value is None:
            return Extraction(field, reason="tagger_span_not_normalizable", start=s, end=e, text=text,
                              source="tagger")
        return Extraction(field, value, op, s, e, text, "tagger", conf)
