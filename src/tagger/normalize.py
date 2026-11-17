"""Span -> canonical value for the fine-tuned tagger, deterministic, on top of src/extraction.

    norm = SpanNormalizer.from_artifacts(ARTIFACTS)      # lexicon, numeric words, common words, taxonomy
    norm.value("נדל״ן/מחיר", "עד 2 מיליון", "max")      # -> {"max": 2000000}
    norm.assemble(domain, [(field, span_text, op), ...], implied={field: value})  -> {"category", "params"}

Numeric fields: Hebrew number reader (`parse_numeric`); the operator comes from the model when it is a numeric op.
Booleans: polarity from the operator (true / false). Everything else: the field-local lexicon / fuzzy / cue /
date matchers of FieldExtractor run on the span text alone. Several spans of one field form a list.
Taxonomy rules then fill sector / manufacturer / model from sub-category / model / sub-model.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.extraction.extractor import FieldExtractor
from src.extraction.model import ExtractorModel
from src.extraction import numbers as _nums
from src.extraction.numbers import STORAGE_FIELDS, _token_value, parse_numeric, read_values
from src.extraction.taxonomy import TaxonomyRules
from src.extraction.text import fold, tokens
from src.extraction.types import Extraction

LEXICON = "lexicon_v2.json"           # the tagger's field lexicon (built from its train)
WORDY = "numeric_words.json"          # (field, folded span text) -> value, for numeric spans with no number
COMMON = "common_words.json"          # words frequent in train queries: never treated as number typos
TAXONOMY = "taxonomy_v2.json"
NUMERIC_OPS = ("exact", "min", "max", "range")

# F1: a storage unit glued to its number («1TB», «512GB») is split so the number reader sees both
_GLUED_UNIT = re.compile(r"(\d)(?=(?:TB|GB|Tb|Gb|tb|gb|טרה|גיגה|ג׳יגה|ג'יגה)(?![A-Za-z]))")
# F3: generic property types, dropped when a more specific type of the same query is also present
GENERIC_PROPERTY_TYPES = frozenset({"דירה", "בית פרטי/וילה"})

# ---------------------------------------------------------------- number-word typos (span text only)
_FINAL = str.maketrans({"מ": "ם", "נ": "ן", "פ": "ף", "כ": "ך", "צ": "ץ"})
_NUMBER_WORDS = sorted({*_nums._UNITS, *_nums._TENS, *_nums._HUNDREDS, *_nums._THOUSANDS_FIXED, *_nums._MULT,
                        *_nums._FRACTIONS, "מאות"} - {"מיל"})
_BOUND = {"עד", "מ", "בין", "לפחות", "מעל", "מתחת", "פחות", "יותר", "מינימום", "מקסימום", "ל"}
_HEB = set("אבגדהוזחטיכךלמםנןסעפףצץקרשת")


def _edit1(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def _correct_word(w: str, common: frozenset = frozenset()) -> str | None:
    """A misspelt number word -> its spelling (keeps a one-letter prefix), or None."""
    for pre in ("",) + tuple(p for p in "ובלמש" if w.startswith(p) and len(w) > 3):
        core = w[len(pre):]
        cand = core.translate(_FINAL) if core and core[-1] in "מנפכצ" else core
        if cand != core and _token_value(cand) is not None:
            return pre + cand
        if len(core) > 2 and core[-1] == core[-2] and _token_value(core[:-1]) is not None:
            return pre + core[:-1]
        if len(core) >= 4 and w not in common:   # edit-distance guesses never touch frequent words
            hits = [n for n in _NUMBER_WORDS if len(n) >= 4 and _edit1(core, n)]
            if len(hits) == 1:
                return pre + hits[0]
    return None


def fix_number_typos(text: str, common: frozenset = frozenset()) -> str:
    """Fix misspelt number words in a numeric span: «עד עשרימ אלפ» -> «עד עשרים אלף», «חמשים אלף» -> «חמישים אלף».
    A word is only rewritten when the reader does not know it and the rewrite sits next to a number, a multiplier or
    a bound word, so ordinary words («שנים», «חדש») are left alone."""
    words = text.split(" ")
    known = [_token_value(w) is not None for w in words]
    out = list(words)
    for i, w in enumerate(words):
        if known[i] or not w or not set(w) <= _HEB:
            continue
        fixed = _correct_word(w, common)   # «שינה» is a word, not «שישה»
        if fixed is None:
            continue
        near = [j for j in (i - 1, i + 1) if 0 <= j < len(words)]
        if any(known[j] or words[j] in _BOUND or _correct_word(words[j], common) for j in near):
            out[i] = fixed
    return " ".join(out)


_POW2 = {125: 128, 250: 256, 500: 512, 1000: 1024, 2000: 2048}


def _storage(v):
    """Storage sizes are powers of two: «חמש מאות ג׳יגה» = 512."""
    if isinstance(v, dict):
        return {k: _POW2.get(x, x) for k, x in v.items()}
    return _POW2.get(v, v)


def _loose(s: str) -> str:
    return re.sub(r"[\s\-־–]+", " ", fold(s)).strip().lower()


def _numbers(v) -> list:
    if isinstance(v, dict):
        return [v[k] for k in ("min", "max") if k in v]
    return [v] if isinstance(v, (int, float)) and not isinstance(v, bool) else []


def reshape(value, op: str):
    """Put the numbers parse_numeric read into the shape of the model's operator."""
    nums = _numbers(value)
    if not nums or op not in NUMERIC_OPS:
        return value
    if op == "range":
        return {"min": min(nums), "max": max(nums)} if len(nums) >= 2 else value
    if op == "min":
        return {"min": nums[0]}
    if op == "max":
        return {"max": nums[-1]}
    return nums[0] if len(nums) == 1 else value


class SpanNormalizer:
    def __init__(self, extractor: FieldExtractor, taxonomy: dict, wordy: dict | None = None,
                 common: frozenset = frozenset()):
        self.ex = extractor
        self.rules = TaxonomyRules(taxonomy)
        self.wordy = wordy or {}
        self._common = frozenset(common)   # real words, never number typos («שינה» is not «שישה»)
        self.canon = {}   # field -> folded canonical value -> value, for spans the matchers miss ("6" = Mazda 6)
        counts = defaultdict(Counter)   # field -> canonical value -> how often train labels use it
        for f, fi in extractor.model.fields.items():
            for vals in fi.surfaces.values():
                for v, n in vals.items():
                    v = json.loads(v)
                    if isinstance(v, str):
                        self.canon.setdefault(f, {})[fold(v).lower()] = v
                        counts[f][v] += n
        # F2: canonical names up to hyphens / spaces («נאות-כרמים» = «נאות כרמים»); when the labels themselves use
        # both spellings, the more frequent one wins
        self.loose = {}
        for f, c in counts.items():
            for v, _ in sorted(c.items(), key=lambda kv: kv[1]):
                self.loose.setdefault(f, {})[_loose(v)] = v

    @classmethod
    def from_artifacts(cls, directory: Path) -> "SpanNormalizer":
        d = Path(directory)
        tax = json.loads((d / TAXONOMY).read_text(encoding="utf-8"))
        model = ExtractorModel.load(d / LEXICON)
        return cls(FieldExtractor(model, TaxonomyRules(tax)), tax, json.loads((d / WORDY).read_text(encoding="utf-8")),
                   frozenset(json.loads((d / COMMON).read_text(encoding="utf-8"))))

    def ftype(self, field: str) -> str | None:
        fi = self.ex.model.fields.get(field)
        return fi.type if fi else None

    def value(self, field: str, text: str, op: str | None = None, trust_op: bool = True):
        t = self.ftype(field)
        if t is None:
            return None
        if t == "bool":
            if op in ("true", "false"):
                return op == "true"
            e = self.ex.extract(text, field)
            return e.value if e.found else True
        if t == "numeric":
            if field.split("/", 1)[1] in STORAGE_FIELDS:
                text = _GLUED_UNIT.sub(lambda m: m.group(1) + " ", text)
            text = fix_number_typos(text, self._common)
            p_op, v = parse_numeric(text, field)
            if v is not None and field.split("/", 1)[1] in STORAGE_FIELDS and not any(ch.isdigit() for ch in text):
                v = _storage(v)   # spelt-out sizes only: «500 גיגה» stays 500
            if v is None:
                v = self.wordy.get(field + "	" + fold(text).strip())
                if v is None:
                    return {"min": 1} if op == "min" else 1 if op == "exact" else None
                return v
            return reshape(v, op) if trust_op and op in NUMERIC_OPS and op != p_op else v
        e = self.ex.extract(text, field)
        if e.found:
            return self.canonical(field, e.value)
        c = self.canon.get(field, {}).get(fold(text).strip().lower())
        if c is not None:
            return c
        if t == "text":   # open vocabulary: the span itself, minus a glued one-letter prefix the lexicon would drop
            toks = tokens(text)
            return self.canonical(field, text.strip()) if toks else None
        return None

    def canonical(self, field: str, value):
        """F2: a name equal to a known canonical value up to hyphens, spaces or a glued one-letter prefix takes that
        canonical value («נאות-כרמים» -> «נאות כרמים», «בעין גנים» -> «עין גנים»). Anything else is returned as is."""
        if not isinstance(value, str):
            return value
        table = self.loose.get(field, {})
        key = _loose(value)
        if key in table:
            return table[key]
        if len(key) > 3 and key[0] in "בהולמ" and key[1:] in table:
            return table[key[1:]]
        return value

    def enum_value(self, field: str, text: str, head: tuple | None, policy: str):
        """Categorical value under a value policy. head = (value, probability) from the learned value head.
            lexicon  matchers only (the head is ignored)
            head     the head's value
            hybrid   matchers; the head only where they find nothing
            confident  the head when its probability >= 0.9, else the matchers, else the head"""
        if head is None or policy == "lexicon":
            return self.value(field, text)
        hv, hp = head
        if policy == "head" or (policy == "confident" and hp >= 0.9):
            return hv
        v = self.value(field, text)
        return v if v is not None else hv

    def _query_fallback(self, query: str, field: str, near: tuple[int, int] | None, window: int = 12):
        """Frozen Fusion's field-local extractor on the whole query, for one field the tagger says is there. With
        `near`, the value must come from within `window` chars of the tagger's span (a partial span «דול» of
        «דול פין», a glued «מאובזרתבמלוא»)."""
        e = self.ex.extract(query, field)
        if not e.found:
            return None
        if near is not None and e.start is not None and not (e.start < near[1] + window and e.end > near[0] - window):
            return None
        return e.value

    def assemble(self, domain: str | None, spans: list[tuple],
                 implied: dict[str, object] | None = None, trust_op: bool = True,
                 value_policy: str = "lexicon", query: str | None = None, fallback: str = "none",
                 present_without_span: set[str] | frozenset = frozenset()) -> dict | None:
        """spans: (field, span_text, op[, (head_value, head_prob)[, start, end]]) for one query.
        fallback (needs `query`):
            none      span text only
            span      a span whose text yields no value retries the field-local extractor near the span
            presence  `span`, plus the extractor on the whole query for non-numeric fields the presence head
                      predicts but no span covers (`present_without_span`)
        Returns {"category", "params"} or None for a non-search."""
        if domain in (None, "none"):
            return None
        vals: dict[str, list] = {}
        for sp in spans:
            f, text, op = sp[:3]
            if not f.startswith(domain + "/"):
                continue
            if self.ftype(f) == "enum":
                v = self.enum_value(f, text, sp[3] if len(sp) > 3 else None, value_policy)
            else:
                v = self.value(f, text, op, trust_op)
            if v is None and fallback != "none" and query is not None and self.ftype(f) != "numeric":
                v = self._query_fallback(query, f, (sp[4], sp[5]) if len(sp) > 5 else None)
            if v is not None and not any(v == x for x in vals.get(f, [])):
                vals.setdefault(f, []).append(v)
        if fallback == "presence" and query is not None:
            for f in present_without_span:
                if f.startswith(domain + "/") and f not in vals and self.ftype(f) not in (None, "numeric"):
                    v = self._query_fallback(query, f, None)
                    if v is not None:
                        vals[f] = [v]
        # F3: one property type, the most specific: «דירת גג» -> גג/טיפוס, not [דירה, גג/טיפוס]
        types = vals.get("נדל״ן/סוגי_נכס")
        if types and len(types) > 1 and any(v not in GENERIC_PROPERTY_TYPES for v in types):
            vals["נדל״ן/סוגי_נכס"] = [v for v in types if v not in GENERIC_PROPERTY_TYPES]
        for f, vs in vals.items():   # "מ-75 מ״ר ... עד 100 מ״ר": two one-sided bounds of one field form a range
            if len(vs) == 2 and all(isinstance(v, dict) and len(v) == 1 for v in vs) \
                    and {k for v in vs for k in v} == {"min", "max"}:
                vals[f] = [{**vs[0], **vs[1]}]
        out = {}
        for f, vs in vals.items():
            if len(vs) == 1 or not all(isinstance(v, str) for v in vs):
                out[f] = Extraction(f, vs[0], source="span")
            else:
                out[f] = Extraction(f, vs, source="span")
        for f, v in (implied or {}).items():
            if f.startswith(domain + "/") and f not in out and v is not None:
                out[f] = Extraction(f, v, source="implied")
        # taxonomy needs the parent present to derive into it: request every parent that has a child
        for child, parent in (("יד_שנייה/תתי_קטגוריות", "יד_שנייה/סקטורים"), ("רכב/תתי_דגמים", "רכב/דגמים"),
                              ("רכב/דגמים", "רכב/יצרנים")):
            if child in out and parent not in out:
                out[parent] = Extraction(parent)
        self.rules.apply(domain, out)
        params = {f.split("/", 1)[1]: e.value for f, e in out.items() if e.found}
        if self.hybrid_vehicle_type:
            params = hybrid_vehicle_type(domain, params)
        return {"category": domain, "params": params}

    hybrid_vehicle_type = False   # product rule, off by default (the dataset labels do not follow it; see below)


HYBRID_FUELS = frozenset({"היברידי", "היברידי נטען"})
HYBRID_VEHICLE_TYPE = "רכב היברידי/חשמלי"   # the taxonomy's value for רכב/סוגי_רכב


def hybrid_vehicle_type(domain: str | None, params: dict) -> dict:
    """Product rule: a hybrid powertrain also sets the vehicle type. «קאמרי היברידי» -> סוג_דלק: היברידי and
    סוגי_רכב: רכב היברידי/חשמלי. An explicit vehicle type (פרטי, רכב שטח, ...) is never overwritten.

    Off by default (SpanNormalizer.hybrid_vehicle_type), because the dataset convention is the opposite: «a powertrain
    named in the query is סוג_דלק», followed by 92% of hybrid rows in train. Turning it on lowers JSON exact
    against the current labels; it should go on together with the same change in the generator's labelling rule."""
    if domain != "רכב" or "סוגי_רכב" in params:
        return params
    fuel = params.get("סוג_דלק")
    fuels = fuel if isinstance(fuel, list) else [fuel]
    if any(f in HYBRID_FUELS for f in fuels):
        return {**params, "סוגי_רכב": HYBRID_VEHICLE_TYPE}
    return params
