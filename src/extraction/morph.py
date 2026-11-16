"""Optional Hebrew morphology backend (Stanza): prefix segmentation + lemmas.

Stanza's Hebrew pipeline splits multi-word tokens («ברחובות» -> ב + רחובות, «וכסוף» -> ו + כסוף) and lemmatizes.
The extractor uses the segmented forms and lemmas as extra matching variants for a token, and the tagger as extra
features. Off by default; requires `pip install stanza` and a one-time `stanza.download("he")`.
"""
from __future__ import annotations

from functools import lru_cache

from .text import tokens

FUNCTION_WORDS = {"ב", "ה", "ל", "ו", "מ", "ש", "כ", "את", "של", "על", "_ה_", "ה_"}


class HebrewMorph:
    def __init__(self):
        import stanza  # noqa: F401  (fail early with a clear ImportError if missing)

        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            import stanza

            self._pipe = stanza.Pipeline("he", processors="tokenize,mwt,pos,lemma", verbose=False,
                                         download_method=None)
        return self._pipe

    @lru_cache(maxsize=20000)
    def _analyze(self, query: str) -> tuple:
        doc = self.pipe(query)
        out = []
        for sent in doc.sentences:
            for tok in sent.tokens:
                forms = [w.text for w in tok.words if w.text not in FUNCTION_WORDS]
                lemmas = [w.lemma for w in tok.words if w.lemma and w.lemma not in FUNCTION_WORDS]
                out.append((tok.start_char, tok.end_char, tuple(forms), tuple(lemmas)))
        return tuple(out)

    def variants(self, query: str) -> dict[int, list[str]]:
        """Extractor-token start offset -> segmented word forms and lemmas of that token (prefixes removed)."""
        analysed = self._analyze(query)
        out: dict[int, list[str]] = {}
        for t in tokens(query):
            for s, e, forms, lemmas in analysed:
                if s <= t.start < e or t.start <= s < t.end:
                    vals = [v for v in (*forms, *lemmas) if v and v != t.text]
                    if vals:
                        out.setdefault(t.start, [])
                        out[t.start] += [v for v in vals if v not in out[t.start]]
        return out
