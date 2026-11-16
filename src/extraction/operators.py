"""Provisional, span-local operator detector.

detect_operator(local_span_text) looks only at ONE field's span text, never at the whole query: a query can mix
operators ("דירת 4 חדרים עד 2 מיליון" = rooms exact, price max). It is a stand-in until the span/value extractor
returns operators. Cues were developed on train spans only; accuracy on val_eval is reported by the runner.
Only bounded-vs-exact matters for picking a point/range field name; min/max/range confusions do not.
"""
from __future__ import annotations

import re

# Hebrew letters count as word characters, so (?<!\w) / (?!\w) act as word boundaries around Hebrew cues.
_B, _E = r"(?<!\w)", r"(?!\w)"
_NUM = r"(\d[\d,.]*|אחד|אחת|שתיים|שתי|שניים|שני|שלוש|ארבע|חמש|שש|שבע|שמונה|תשע|עשר)"

EXACT = [r"בדיוק"]  # 'בדיוק 190 מ״ר, לא פחות ולא יותר' is exact despite the bound words
RANGE = [
    _B + r"בין" + _E,
    r"\d\s*[-–]\s*\d",
    _NUM + r"\S*\s+(ו?עד|ל-?)\s*" + _NUM,  # '1 עד 3', '0 עד 1'
    r"^מ-?\s*" + _NUM + r".*\sו?עד",  # 'מ-4,900 ש"ח עד 7,000' (a number right after the מ prefix)
    r"^מיד\s+\S+\s+ו?עד",  # 'מיד אחת ועד שלוש'
]
MIN_NEGATED = [_B + r"לא\s+פחות\s+מ", r"אין\s+מצב\s+פחות", _B + r"לפחות"]  # checked before MAX ('פחות מ')
MAX = [_B + r"[ושכ]?לא" + _E + r".{0,20}יותר\s*מ", r"לכל\s+היותר", r"מקסימום", r"תקרה", _B + r"לא" + _E + r".{0,12}מעל",
       r"לא\s+(עולה|עובר|תעבור|יעבור)", r"שלא\s+(עולה|עובר)", r"שלא\s+עבר", _B + r"פחות\s+מ", r"מתחת\s+ל", r"ומטה",
       _B + r"ו?עד(?=\s|\d|$)"]
MIN = [r"ומעל", r"והלאה", _B + r"[ו]?מ-?(?=\d)", r"מינימום", r"החל\s+מ", _B + r"מעל" + _E, _B + r"יותר\s+מ", r"^מ-\s*\S", r"^מ\s*\d",
       r"^מ-?\s*" + _NUM]

OPERATORS = ("exact", "min", "max", "range")


def detect_operator(local_span_text: str) -> str:
    t = local_span_text.strip()
    for op, patterns in (("exact", EXACT), ("range", RANGE), ("min", MIN_NEGATED), ("max", MAX), ("min", MIN)):
        if any(re.search(p, t) for p in patterns):
            return op
    return "exact"


def is_bounded(operator: str) -> bool:
    return operator in ("min", "max", "range")
