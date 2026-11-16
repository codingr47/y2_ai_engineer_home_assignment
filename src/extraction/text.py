"""Hebrew text helpers shared by the extractor: normalization for matching, tokens with offsets, prefixes."""
from __future__ import annotations

import re
from dataclasses import dataclass

INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠‎‏﻿‪‫‬ "), " ")
FINALS = str.maketrans("ךםןףץ", "כמנפצ")
QUOTES = str.maketrans({"״": '"', "”": '"', "“": '"', "׳": "'", "’": "'", "`": "'", "־": "-", "–": "-", "—": "-"})
PREFIX_LETTERS = "ובלמהכש"  # one-letter Hebrew prefixes: ו ב ל מ ה כ ש (and combos like וב, מה, שב)
NEGATIONS = ("בלי", "ללא", "לא", "אין", "בלא", "מבלי", "בלתי")
CORRECTIONS = ("רגע", "בעצם", "סליחה", "כלומר", "טעות")
_TOKEN = re.compile(r"[^\s,;:!?()\[\]{}]+")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def clean(text: str) -> str:
    """Display-preserving cleanup: invisible marks -> space, quote variants unified."""
    return text.translate(INVISIBLE).translate(QUOTES)


def fold(text: str) -> str:
    """Matching key: cleaned, final letters folded, quotes/periods/hyphens dropped, spaces squeezed, lowercased."""
    t = clean(text).translate(FINALS).lower()
    t = re.sub(r"[\"'.]", "", t)
    t = re.sub(r"[\-_/]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)  # 3+ repeated letters collapse (legit doubles like vav-vav stay)
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str) -> list[Token]:
    """Whitespace/punctuation tokens with char offsets into the ORIGINAL text (offsets survive clean())."""
    cleaned = clean(text)  # same length as text: translate maps 1 char -> 1 char
    out = []
    for m in _TOKEN.finditer(cleaned):
        tok = m.group(0).strip(".\"'-")
        if not tok:
            continue
        lead = m.group(0).find(tok)
        out.append(Token(tok, m.start() + lead, m.start() + lead + len(tok)))
    return out


def prefix_variants(word: str, max_prefix: int = 2) -> list[str]:
    """'ובתל' -> ['ובתל', 'בתל', 'תל']: the word with up to two leading one-letter prefixes removed."""
    out, w = [word], word
    for _ in range(max_prefix):
        if len(w) > 2 and w[0] in PREFIX_LETTERS:
            w = w[1:].lstrip("-")
            out.append(w)
        else:
            break
    return out


def edit1(a: str, b: str) -> bool:
    """True if a and b differ by one insertion, deletion, substitution or adjacent swap."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def negated_before(query_tokens: list[Token], start_index: int, window: int = 2) -> bool:
    """Is one of the `window` tokens before position `start_index` a negation word (or prefixed one)?"""
    for t in reversed(query_tokens[max(0, start_index - window):start_index]):
        variants = prefix_variants(t.text)
        if any(v in CORRECTIONS for v in variants):  # "לא, רגע ממ״ד": the correction cancels the negation
            return False
        if any(v in NEGATIONS for v in variants):
            return True
    return False
