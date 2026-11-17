"""Runtime query cleanup before the tagger: fix mechanical typos.

    cleaner = QueryCleaner.from_file(ARTIFACTS / "clean_vocab.json")
    cleaner.clean("מאזdה בחולוןן עד מיל,יון")   # -> "מאזדה בחולון עד מיליון"; cleaner.log lists the rewrites

A word is rewritten only when a candidate is clearly the more frequent word: freq >= RATIO * freq(original) +
MIN_GAIN, with frequencies from train queries + lexicon surface words. All rules except final_form also require the
original to be rare as written (<= RARE occurrences), so real words («כלל», «לפטופ», «מוכנים») stay.
    invisible    zero-width / direction marks removed                          «ב​בקשה»
    final_form   non-final letter at the end, or final letter inside a word     «אלפ» -> «אלף», «מחףש» -> «מחפש»
    repeat       a doubled final letter at the end (never Hebrew), a doubled    «בחולוןן», «חדשש», «דחוףףף»
                 end letter of an unseen word, or a letter 3+ times in a row
    lookalike    one Latin letter or digit inside a Hebrew word                 «מאזdה», «טלפ1ן»
    punct        . , _ * inside a word (never ' " ׳ ״: abbreviations, geresh)    «יונ.דאי», «מיל,יון»
    split        two glued known words, both at least 3 letters                «טויוטהקורולה», «במצבטוב»
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


HEB = set("אבגדהוזחטיכךלמםנןסעפףצץקרשת")
LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"
FINALS = "ךםןףץ"
TO_FINAL = str.maketrans("כמנפצ", FINALS)
FROM_FINAL = str.maketrans(FINALS, "כמנפצ")
PREFIX = "ובלמהכש"
RATIO, MIN_GAIN, RARE, SPLIT_MIN = 5, 5, 3, 20
VOCAB = "clean_vocab.json"     # word -> frequency (train queries + lexicon surfaces), built by training_bert
_WORD = re.compile(r"\S+")
_PUNCT = re.compile(r"[.,_*]")
_KEEP = set("'\"׳״-")
_INVISIBLE = re.compile("[​-‏‪-‮⁠﻿]")
_EDGE = ".,!?:;()\"'"


def _heb_count(w: str) -> int:
    return sum(ch in HEB for ch in w)


def _finalize(w: str) -> str:
    return w[:-1] + w[-1].translate(TO_FINAL) if w else w


@dataclass
class QueryCleaner:
    freq: dict[str, int]
    log: list = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path) -> "QueryCleaner":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def f(self, w: str) -> int:
        """Frequency of a word, also behind a one-letter prefix («בחולון» counts as «חולון»)."""
        n = self.freq.get(w, 0)
        if len(w) > 3 and w[0] in PREFIX:
            n = max(n, self.freq.get(w[1:], 0))
        return n

    def candidates(self, w: str, rare: bool) -> list[tuple[str, str, int]]:
        out = []
        if w[-1] in "כמנפצ":
            out.append(("final_form", _finalize(w)))
        inner = w[:-1].translate(FROM_FINAL) + w[-1]
        if inner != w:
            out.append(("final_form", inner))
        if re.search(f"([{FINALS}])\\1+$", w):   # a doubled final letter is never Hebrew, however often it occurs
            out.append(("repeat", re.sub(f"([{FINALS}])\\1+$", "\\1", w)))
        if rare:
            r = w
            if self.f(w) == 0:
                r = re.sub(r"(.)\1+$", "\\1", r)
            r = re.sub(r"(.)\1{2,}", "\\1", r)
            if r != w:
                out.append(("repeat", _finalize(r)))
            odd = [i for i, ch in enumerate(w) if ch not in HEB and ch not in _KEEP and not _PUNCT.match(ch)]
            if len(odd) == 1 and _heb_count(w) >= 3:
                i = odd[0]
                out += [("lookalike", _finalize(w[:i] + ch + w[i + 1:])) for ch in LETTERS]
            if _PUNCT.search(w[1:-1]):
                out.append(("punct", _finalize(w[0] + _PUNCT.sub("", w[1:-1]) + w[-1])))
        scored = [(k, c, self.f(c)) for k, c in out]
        if rare and len(w) >= 6 and all(ch in HEB for ch in w):
            for i in range(3, len(w) - 2):
                a, b = _finalize(w[:i]), w[i:]
                fa, fb = self.f(a), self.f(b)
                if fa >= SPLIT_MIN and fb >= SPLIT_MIN:
                    scored.append(("split", f"{a} {b}", min(fa, fb)))
        return scored

    def fix_word(self, w: str) -> tuple[str, str | None]:
        core = w.strip(_EDGE)
        if len(core) < 3 or _heb_count(core) < 2:
            return w, None
        base = self.f(core)
        best = max(self.candidates(core, base <= RARE), key=lambda x: x[2], default=None)
        if best is None or best[2] < RATIO * base + MIN_GAIN:
            return w, None
        return w.replace(core, best[1], 1), best[0]

    def clean(self, query: str) -> str:
        stripped = _INVISIBLE.sub("", query)
        kinds = [("invisible", query, stripped)] if stripped != query else []
        parts, pos = [], 0
        for m in _WORD.finditer(stripped):
            parts.append(stripped[pos:m.start()])
            new, kind = self.fix_word(m.group())
            parts.append(new)
            if kind:
                kinds.append((kind, m.group(), new))
            pos = m.end()
        parts.append(stripped[pos:])
        self.log = kinds
        return "".join(parts)
