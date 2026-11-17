"""Training-side query cleaner: builds the word-frequency table, then uses the shared runtime cleaner.

    cleaner = QueryCleaner.load()   # builds training/training_bert/artifacts/clean_vocab.json from train if missing
The cleanup rules live in src/tagger/clean.py, shared with the runtime parser. The data files are never touched.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.tagger import clean as _rt
from src.tagger.clean import _EDGE, _INVISIBLE  # noqa: F401

from .data import ARTIFACTS, load

VOCAB = ARTIFACTS / _rt.VOCAB


def build_vocab(path: Path = VOCAB) -> Path:
    """Word frequencies from train queries (edge punctuation stripped) + lexicon surface words."""
    c = Counter()
    for r in load("train"):
        for w in _INVISIBLE.sub("", r["query"]).split():
            w = w.strip(_EDGE)
            if w:
                c[w] += 1
    lex = ARTIFACTS / "lexicon_v2.json"
    if lex.exists():
        d = json.loads(lex.read_text(encoding="utf-8"))
        for fi in (d.get("fields") or {}).values():
            for surf in (fi.get("surfaces") or {}):
                for w in surf.split():
                    c[w] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    return path


class QueryCleaner(_rt.QueryCleaner):
    @classmethod
    def load(cls, path: Path = VOCAB) -> "QueryCleaner":
        if not path.exists():
            build_vocab(path)
        return cls.from_file(path)
