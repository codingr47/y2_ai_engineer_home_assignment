"""Field-local value extraction: (query, field) -> Extraction, and pipeline helpers.

    ex = FieldExtractor.load(ARTIFACTS)                     # lexicon only
    ex = FieldExtractor.load(ARTIFACTS, use_tagger=True)    # hybrid: + learned span tagger fallback
    ex = FieldExtractor.load(ARTIFACTS, use_morph=True)     # + optional Hebrew morphology (stanza)
    ex.extract("מחפש דירת 4 חדרים עד 2 מיליון", "נדל״ן/מחיר")   # -> value {"max": 2000000}, operator "max"
    ex.extract_all(query, "נדל״ן", ["נדל״ן/מס׳_חדרים", "נדל״ן/מחיר"])
"""
from __future__ import annotations

from pathlib import Path

from .match import (bool_match, combine_list, cue_text_match, date_matches, fuzzy_matches, lexicon_matches,
                    numeric_candidates, numeric_score, pick_date, unit_enum_match)
from .model import LOCATION_FIELDS, ExtractorModel
from .taxonomy import TaxonomyRules
from .text import tokens
from .types import Extraction

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


class FieldExtractor:
    def __init__(self, model: ExtractorModel, rules: TaxonomyRules | None = None, tagger=None, morph=None):
        self.model = model
        self.rules = rules
        self.tagger = tagger
        self.morph = morph
        self._k = {d: sum(1 for f, fi in model.fields.items() if f.startswith(d + "/") and fi.type == "numeric")
                   for d in model.cue_totals}
        self._units = frozenset(model.unit_tokens)
        self._common = frozenset(model.common_tokens)

    @classmethod
    def load(cls, artifacts: Path = ARTIFACTS, use_tagger: bool = False, use_morph: bool = False,
             taxonomy: dict | None = None) -> "FieldExtractor":
        model = ExtractorModel.load(Path(artifacts) / "lexicon_v2.json")
        rules = TaxonomyRules(taxonomy) if taxonomy is not None else TaxonomyRules.from_artifacts(artifacts)
        tagger = None
        if use_tagger:
            from .tagger import SpanTagger
            tagger = SpanTagger.load(Path(artifacts) / "tagger.joblib")
        morph = None
        if use_morph:
            from .morph import HebrewMorph
            morph = HebrewMorph()
        return cls(model, rules, tagger, morph)

    # ------------------------------------------------------------------ one field
    def extract(self, query: str, field: str, _ctx: dict | None = None) -> Extraction:
        ctx = _ctx if _ctx is not None else self._context(query, field.split("/", 1)[0])
        fi = self.model.fields.get(field)
        if fi is None:
            return Extraction(field, reason="unknown_field")
        q, qtok, morph = query, ctx["tokens"], ctx["morph"]
        m = None
        if fi.type == "numeric":
            ranked = self.numeric_ranked(field, ctx)
            if ranked:
                score, cand, op, value = ranked[0]
                return Extraction(field, value, op, cand.start, cand.end, cand.text, "numeric", score)
        elif fi.type == "bool":
            m = bool_match(fi, q, qtok, morph)
        elif fi.type == "date":
            m = pick_date(field, q, ctx["dates"])
        else:
            m = combine_list(fi, q, lexicon_matches(fi, q, qtok, morph))
            if m is None and fi.type == "text":
                m = cue_text_match(field, q, self._common)
            if m is None:
                m = unit_enum_match(fi, q)
            if m is None:
                m = combine_list(fi, q, fuzzy_matches(fi, q, qtok, morph))
        if m is not None:
            return Extraction(field, m.value, None, m.start, m.end, m.text, m.source, m.score)
        if self.tagger is not None:
            t = self.tagger.extract(query, field, fi, self.model, ctx)
            if t is not None:
                return t
        return Extraction(field, reason="no_span")

    def numeric_ranked(self, field: str, ctx: dict):
        fi = self.model.fields[field]
        dom = field.split("/", 1)[0]
        out = []
        for cand in ctx["numeric"]:
            score, op, value = numeric_score(fi, cand, ctx["query"], ctx["tokens"], self.model.cue_totals.get(dom, {}),
                                             max(1, self._k.get(dom, 1)))
            if value is not None:
                out.append((score, cand, op, value))
        return sorted(out, key=lambda x: -x[0])

    def _context(self, query: str, domain: str | None) -> dict:
        qtok = tokens(query)
        morph = self.morph.variants(query) if self.morph is not None else None
        return {"query": query, "domain": domain, "tokens": qtok, "morph": morph, "lemmas": morph,
                "numeric": numeric_candidates(query, qtok, self._units, domain), "dates": date_matches(query)}

    # ------------------------------------------------------------------ several fields of one query
    def extract_all(self, query: str, domain: str | None, fields: list[str]) -> dict[str, Extraction]:
        """Field-local extraction for each field, then deterministic cross-field resolution:
        two numeric fields cannot claim the same number, location fields cannot share a span, and taxonomy
        derivations fill sector / manufacturer from sub-category / model."""
        ctx = self._context(query, domain)
        out = {f: self.extract(query, f, ctx) for f in fields}
        self._resolve_numeric(out, ctx)
        self._resolve_locations(out)
        if self.rules is not None:
            self.rules.apply(domain, out)
        return out

    def _resolve_numeric(self, out: dict[str, Extraction], ctx: dict):
        num = [f for f, e in out.items() if f in self.model.fields and self.model.fields[f].type == "numeric"]
        if len(num) < 2:
            return
        options = {f: self.numeric_ranked(f, ctx) for f in num}
        pairs = sorted(((s, f, c, op, v) for f in num for s, c, op, v in options[f]), key=lambda x: -x[0])
        taken, done = [], set()
        for s, f, c, op, v in pairs:
            if f in done or any(c.start < t.end and c.end > t.start for t in taken):
                continue
            out[f] = Extraction(f, v, op, c.start, c.end, c.text, "numeric", s)
            taken.append(c)
            done.add(f)
        for f in num:
            if f not in done and out[f].source == "numeric":
                out[f] = Extraction(f, reason="number_claimed_by_other_field")

    @staticmethod
    def _resolve_locations(out: dict[str, Extraction]):
        loc = [e for f, e in out.items() if f.split("/", 1)[-1] in LOCATION_FIELDS and e.found and e.start is not None]
        loc.sort(key=lambda e: -e.score)
        kept = []
        for e in loc:
            if any(e.start < k.end and e.end > k.start for k in kept):
                out[e.field] = Extraction(e.field, reason="span_claimed_by_other_location")
            else:
                kept.append(e)

    @staticmethod
    def to_json(domain: str | None, extractions: dict[str, Extraction]) -> dict | None:
        """{"category", "params"} with found values only; None for a non-search (domain None / 'none')."""
        if domain in (None, "none"):
            return None
        return {"category": domain,
                "params": {f.split("/", 1)[1]: e.value for f, e in extractions.items() if e.found}}
