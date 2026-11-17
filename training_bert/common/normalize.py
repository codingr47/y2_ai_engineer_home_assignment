"""Training-side span normalizer: builds the artifacts, then uses the shared runtime normalizer.

    norm = SpanNormalizer.load()     # builds training_bert/artifacts/{lexicon_v2,numeric_words,common_words,
                                     # taxonomy_v2}.json from train if missing, then src.tagger.normalize
The normalization logic itself lives in src/tagger/normalize.py, shared with the runtime parser.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.extraction.model import ExtractorModel
from src.extraction.numbers import _token_value, read_values
from src.extraction.text import fold
from src.tagger import normalize as _rt
from src.tagger.gate import json_equal  # noqa: F401 (re-exported)
from src.tagger.normalize import (HYBRID_FUELS, HYBRID_VEHICLE_TYPE, NUMERIC_OPS, _storage,  # noqa: F401
                                  fix_number_typos, hybrid_vehicle_type, reshape)
from training_frozen_fusion.shared.dataset import load_taxonomy
from training_frozen_fusion.shared.normalize import SCHEMA

from .data import ARTIFACTS, load

LEXICON = ARTIFACTS / _rt.LEXICON
WORDY = ARTIFACTS / _rt.WORDY
COMMON = ARTIFACTS / _rt.COMMON
TAXONOMY = ARTIFACTS / _rt.TAXONOMY


def build_lexicon(path: Path = LEXICON) -> Path:
    rows = load("train")
    tax = load_taxonomy("v2")
    model = ExtractorModel.build(rows, tax, SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    # numeric spans written without a number ("עם חניה", "חניה כפולה", "קומת קרקע"): their most frequent train value
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        for k, ss in (r.get("spans") or {}).items():
            f = f"{r['domain']}/{k}"
            fi = model.fields.get(f)
            if fi is None or fi.type != "numeric" or len(ss) != 1 or read_values(ss[0]["text"]):
                continue
            key = f + "	" + fold(ss[0]["text"]).strip()
            v = json.dumps(r["expected"]["params"][k], ensure_ascii=False, sort_keys=True)
            counts.setdefault(key, {}).setdefault(v, 0)
            counts[key][v] += 1
    WORDY.write_text(json.dumps({k: json.loads(max(c, key=c.get)) for k, c in counts.items()}, ensure_ascii=False),
                     encoding="utf-8")
    build_common_words(rows)
    TAXONOMY.write_text(json.dumps(tax, ensure_ascii=False), encoding="utf-8")
    return path


def build_common_words(rows: list[dict] | None = None) -> Path:
    """Words seen >= 20 times in train queries that are not number words: real words, never number typos."""
    counts = Counter(w for r in (rows if rows is not None else load("train")) for w in r["query"].split())
    COMMON.write_text(json.dumps(sorted(w for w, c in counts.items() if c >= 20 and _token_value(w) is None),
                                 ensure_ascii=False), encoding="utf-8")
    return COMMON




class SpanNormalizer(_rt.SpanNormalizer):
    @classmethod
    def load(cls, lexicon: Path = LEXICON) -> "SpanNormalizer":
        if not lexicon.exists() or not WORDY.exists():
            build_lexicon(lexicon)
        if not COMMON.exists():
            build_common_words()
        if not TAXONOMY.exists():
            TAXONOMY.write_text(json.dumps(load_taxonomy("v2"), ensure_ascii=False), encoding="utf-8")
        return cls.from_artifacts(lexicon.parent)


def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
