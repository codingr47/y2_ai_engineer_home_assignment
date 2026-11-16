"""Hebrew numbers: read the values written in a span and turn them into an operator + value.

Token readers and composition rules are ported from the synthetic generator's verifier
(y2_ai_synthetic_generator/synthetic_generation_app/numeric_text.py, same author), which reads digits, Hebrew
number words, אלף / K / מיליון, וחצי and glued typos. The generator returns *every possible* reading of a span; an
extractor must commit to one, so read_values() segments a span into ordered values instead
("בין 3 ל-4 אלף" -> [3000, 4000]).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .operators import detect_operator

_UNITS = {
    "אפס": 0, "אחד": 1, "אחת": 1, "שתיים": 2, "שתים": 2, "שניים": 2, "שני": 2, "שתי": 2, "שלוש": 3, "שלושה": 3,
    "שלושת": 3, "ארבע": 4, "ארבעה": 4, "ארבעת": 4, "חמש": 5, "חמישה": 5, "חמשה": 5, "חמשת": 5, "שש": 6, "שישה": 6,
    "ששת": 6, "שישת": 6, "שבע": 7, "שבעה": 7, "שבעת": 7, "שמונה": 8, "שמונת": 8, "תשע": 9, "תשעה": 9, "תשעת": 9,
    "עשר": 10, "עשרה": 10, "עשרת": 10,
}
_TENS = {"עשרים": 20, "שלושים": 30, "ארבעים": 40, "חמישים": 50, "שישים": 60, "ששים": 60, "שבעים": 70, "שמונים": 80,
         "תשעים": 90}
_HUNDREDS = {"מאה": 100, "מאתיים": 200}
_ORDINALS = {"ראשון": 1, "ראשונה": 1, "שנייה": 2, "שניה": 2, "שלישי": 3, "שלישית": 3, "רביעי": 4, "רביעית": 4,
             "חמישי": 5, "חמישית": 5, "שישי": 6, "שישית": 6, "שביעי": 7, "שביעית": 7, "שמיני": 8, "שמינית": 8,
             "תשיעי": 9, "תשיעית": 9}
_THOUSANDS_FIXED = {"אלפיים": 2000}
_MULT = {"אלף": 1_000, "אלפים": 1_000, "אלפי": 1_000, "מיליון": 1_000_000, "מליון": 1_000_000,
         "מיליונים": 1_000_000, "מיל": 1_000_000}
_FRACTIONS = {"חצי": 0.5, "רבע": 0.25}
_TEEN = {"עשרה", "עשר"}
_PREFIX = "ובלמהכש"
_DIGITS = re.compile(r"^(\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?)([kK]|[mM])?$")
_SPLIT = re.compile(r"[\s־–—/]+|(?<=\d)(?=[^\d.,kKmM\s])|(?<=[א-ת])(?=\d)|(?<=\d[kKmM])(?=[א-ת])"
                    r"|(?<=[א-ת])-(?=[א-ת])|(?<=[\dkKmM])-(?=\d)")
_CORRECTION = re.compile(r"בעצם|סליחה|רגע|כלומר|טעות|תיקון|(?:הת)?כוונתי")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠‎‏﻿‪‫‬"), None)
_CONNECTOR = re.compile(r"(^|\s)(ו?עד|ל-?|או|-|–)(\s|$)|^ל")
YEAR_FIELDS = {"שנה", "שנת_ייצור"}
STORAGE_FIELDS = {"אחסון_GB", "נפח_אחסון", "זיכרון_RAM"}


@dataclass
class Value:
    value: float
    start: int  # char offsets into the text that was read
    end: int
    has_mult: bool
    words: bool  # composed from number words (not digits)
    mult: float | None = None  # last multiplier in the value (אלף, מיליון, K)


def _token_value(tok: str):
    """('num', v, is_digit) | ('mult', m) | ('hund',) | ('thou', v) | ('half',) | ('teen',) | None"""
    t = tok.strip(".,!?:;()\"'״׳־-–")
    if not t:
        return None
    cands = [t] + ([t[1:].lstrip("-־–")] if len(t) > 1 and t[0] in _PREFIX else [])
    for c in cands:
        if not c:
            continue
        m = _DIGITS.match(c)
        if m:
            raw, suf = m.group(1), (m.group(2) or "")
            raw = raw.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", raw) else raw.replace(",", ".")
            v = float(raw) * (1000 if suf in "kK" and suf else 1_000_000 if suf in "mM" and suf else 1)
            return ("num", v, True)
        for table in (_UNITS, _TENS, _HUNDREDS, _ORDINALS):
            if c in table:
                return ("num", table[c], False)
        if c == "מאות":
            return ("hund",)
        if c in _THOUSANDS_FIXED:
            return ("thou", _THOUSANDS_FIXED[c])
        if c in _MULT:
            return ("mult", _MULT[c])
        if c in _FRACTIONS:
            return ("num", _FRACTIONS[c], False)
        if c == "וחצי" or (c == "חצי" and t.startswith("ו")):
            return ("half",)
        if c in _TEEN:
            return ("teen",)
    for u, uv in _UNITS.items():  # glued teen: «וחמשעשרה» = 15
        if len(u) > 1 and any(t.endswith(u + teen) for teen in _TEEN):
            return ("num", uv + 10, False)
    for i in range(2, len(t) - 2):  # a number word glued to the previous word: «מרפסתאחת», «ידשלישית»
        for table in (_ORDINALS, _UNITS, _TENS):
            if t[i:] in table:
                return ("num", table[t[i:]], False)
    return None


def _combine(vals: list[tuple]) -> float:
    """Hebrew number composition: "אלפיים וארבע מאות חמישים" = 2450, "מיליון וחצי" = 1.5M, "2 מיליון 300 אלף"."""
    total, cur, last, last_mult = 0.0, 0.0, 0.0, None
    for v in vals:
        kind = v[0]
        if kind == "num":
            cur += v[1]
            last, last_mult = v[1], None
        elif kind == "teen":
            cur += 10
        elif kind == "hund":
            cur += (last or 1) * 100 - last
            last = 0.0
        elif kind == "thou":
            total += v[1]
            last_mult = 1000
        elif kind == "mult":
            total += (cur or 1) * v[1]
            cur, last, last_mult = 0.0, 0.0, v[1]
        elif kind == "half":
            if last_mult:
                total += 0.5 * last_mult
            else:
                cur += 0.5
    return total + cur


def _tokens(text: str) -> list[tuple[str, int, int]]:
    out, pos = [], 0
    for m in _SPLIT.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], pos, m.start()))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], pos, len(text)))
    return out


def read_values(text: str) -> list[Value]:
    """Ordered values written in `text` with their offsets. Adjacent digit numbers are separate values; a token
    with a ל- prefix ("ל-4", "לשלוש") starts a new value; number words compose ("שלוש מאות וחמישים")."""
    groups: list[list] = []
    cur: list = []
    for raw, s, e in _tokens(text):
        v = _token_value(raw)
        if v is None:
            if cur:
                groups.append(cur)
            cur = []
            continue
        starts_new = bool(cur) and (
            (v[0] == "num" and v[2] and cur[-1][0][0] in ("num", "teen", "half"))  # "3 4", "שלוש 4"
            or (raw.lstrip("-")[:1] == "ל" and v[0] in ("num", "thou") and len(raw) > 1)  # "ל-4", "לשלוש"
            # "בין אלף חמישים לאלף ארבע מאות": a ל-prefixed multiplier after a number that already has one
            or (raw.lstrip("-")[:1] == "ל" and v[0] == "mult" and len(raw) > 1
                and any(x[0][0] in ("mult", "thou") for x in cur)))
        if starts_new:
            groups.append(cur)
            cur = []
        cur.append((v, s, e))
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        kinds = [x[0] for x in g]
        if not any(k[0] in ("num", "mult", "thou") for k in kinds):
            continue
        mults = [k[1] for k in kinds if k[0] == "mult"]
        out.append(Value(_combine(kinds), g[0][1], g[-1][2], any(k[0] in ("mult", "thou") for k in kinds),
                         not any(k[0] == "num" and k[2] for k in kinds), mults[-1] if mults else None))
    return out


def share_multiplier(vals: list[Value]) -> list[Value]:
    """A multiplier written once for a range applies to both ends: "בין 3 ל-4 אלף", "בין 5.95 ל-6 מיל"."""
    if len(vals) >= 2 and vals[-1].mult and not vals[0].has_mult             and vals[0].value * vals[-1].mult <= vals[-1].value * 1.0001:
        first = vals[0]
        vals = [Value(first.value * vals[-1].mult, first.start, first.end, True, first.words, vals[-1].mult)]             + vals[1:]
    return vals


def _num(v: float):
    return int(v) if float(v).is_integer() else round(v, 4)


def parse_numeric(span_text: str, field: str) -> tuple[str | None, object]:
    """(operator, value) read off one field's span: number for exact, {"min"}/{"max"}/{"min","max"} for bounds."""
    text = span_text.translate(_ZERO_WIDTH).replace(" ", " ")
    corr = list(_CORRECTION.finditer(text))
    vals = read_values(text)
    op_text = text
    if corr:  # self-correction: keep the values (and operator words) written after the last correction word
        after = [v for v in vals if v.start >= corr[-1].end()]
        if after:
            vals, op_text = after, text[corr[-1].end():]
    vals = share_multiplier(vals)
    # two values joined by עד / ל / או / '-' form a range even when the bound words are unusual
    joined = len(vals) >= 2 and (bool(_CONNECTOR.search(text[vals[0].end:vals[1].start]))
                                 or text[vals[1].start:vals[1].start + 1] == "ל")  # "מליון וחצי לשניים וחצי"
    numbers = [v.value for v in vals]
    name = field.split("/", 1)[-1]
    low = text.lower()
    if "דונם" in text:
        numbers = [n * 1000 for n in numbers] or [1000.0]
        if re.search(r"דונם\s+וחצי", text):
            numbers = [n + 500 for n in numbers]
    if name in STORAGE_FIELDS and ("טרה" in text or "tb" in low):
        numbers = [n * 1024 if n < 64 else n for n in numbers] or [1024.0]
    if name == "נפח_מנוע_סמ״ק" and "ליטר" in text:
        numbers = [n * 1000 if n < 10 else n for n in numbers]
    if name in YEAR_FIELDS:
        numbers = [(2000 + n if n <= 30 else 1900 + n) if 0 <= n < 100 and float(n).is_integer() else n
                   for n in numbers]
    if name == "קומה" and "קרקע" in text and not numbers:
        numbers = [0.0]
    if not numbers:
        return None, None
    op = detect_operator(op_text)
    if joined and op != "range" and len(numbers) >= 2:
        op = "range"
    if op == "range" and len(numbers) < 2:  # one value: fall back to the bound word that is present
        op = "max" if re.search(r"(^|\s)ו?עד", op_text) else "min" if re.search(r"(^|\s)מ", op_text) else "exact"
    if op == "range" and len(numbers) >= 2:
        lo, hi = sorted(numbers[:2])
        return "range", {"min": _num(lo), "max": _num(hi)}
    if op == "min":
        return "min", {"min": _num(numbers[0])}
    if op == "max":
        return "max", {"max": _num(numbers[-1] if len(numbers) > 1 and corr else numbers[0])}
    return "exact", _num(numbers[0])
