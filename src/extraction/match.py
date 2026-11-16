"""Deterministic span finders. Each returns candidates with offsets into the original query."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache

from .model import FieldInfo
from .numbers import parse_numeric, read_values
from .text import NEGATIONS, Token, clean, edit1, fold, negated_before, prefix_variants

MAX_NGRAM = 6
# hierarchy levels are usually implied by the item word («טלוויזיה» -> אלקטרוניקה / טלוויזיות); a literal mention
# of another level's value («לסלון») is often a purpose, not the item, so no literal bonus for these
HIERARCHY_FIELDS = {"סקטורים", "תתי_קטגוריות"}
SECOND_HAND = re.compile(r"(?<![א-ת])[ובלמה]?יד\s*[-־]?\s*(?:שנייה|שניה|2)(?![\d.])")
# place names that contain a number word: «בבאר שבע עד 65 אינץ׳» is not 7-65
NUMBERED_PLACES = re.compile(r"(?<![א-ת])[ובלמה]?(?:באר\s*שבע|קרי(?:י)?ת\s*שמונה)(?![א-ת])")
BOUND_LEFT = {"בין", "עד", "מ", "לפחות", "מעל", "מקסימום", "מינימום", "החל", "לא", "יותר", "פחות", "תקרה", "שלא",
              "עובר", "עולה", "מתחת", "בערך", "כ", "בסביבות", "רק", "בדיוק", "כולל"}
BOUND_RIGHT = {"ומעלה", "ומטה", "והלאה", "היותר", "לכל", "מקסימום", "בדיוק", "לפחות", "ומעל", "ופחות"}
BOUND_LEFT |= {"לכל", "היותר", "מקסימלי", "מינימלי"}
# tokens are compared folded (final letters -> regular: «מקסימום» -> «מקסימומ»), so fold the word lists too
BOUND_LEFT = {fold(w) for w in BOUND_LEFT}
BOUND_RIGHT = {fold(w) for w in BOUND_RIGHT}
CONNECTOR = re.compile(r"^\s*(ו?עד|ל-?|או|-|–|ל)\s*$|^\s*-?\s*$")


@dataclass
class Match:
    start: int
    end: int
    text: str
    value: object
    score: float
    source: str


def _best_value(counts: dict[str, int]):
    return json.loads(max(counts.items(), key=lambda kv: kv[1])[0])


def _literal(key: str, value) -> bool:
    """The surface spells its value («i20» -> i20) rather than implying it («של יונדאי» -> a model)."""
    if not isinstance(value, str):
        return False
    v = fold(value)
    return bool(v) and (v == key or v in key or key in v)


# ------------------------------------------------------------------ lexicon (enum / text / derived)
def _ngram_keys(qtok: list[Token], i: int, n: int):
    """Folded n-gram keys, with plain or prefix-stripped first tokens."""
    words = [t.text for t in qtok[i:i + n]]
    firsts = [(w, "plain" if j == 0 else "prefix") for j, w in enumerate(prefix_variants(words[0]))]
    for first, kind in firsts:
        yield fold(" ".join([first] + words[1:])), kind


def lexicon_matches(fi: FieldInfo, query: str, qtok: list[Token]) -> list[Match]:
    """All exact surface matches for the field, longest first, non-overlapping, in query order."""
    cands = []
    for i in range(len(qtok)):
        for n in range(min(MAX_NGRAM, len(qtok) - i), 0, -1):
            for key, kind in _ngram_keys(qtok, i, n):
                if len(key) < 2 or key not in fi.surfaces:
                    continue
                counts = fi.surfaces[key]
                value = _best_value(counts)
                literal = 0.0 if fi.name.split("/", 1)[-1] in HIERARCHY_FIELDS else 2.0 * _literal(key, value)
                score = 1.0 * n + literal + 0.1 * math.log1p(sum(counts.values())) - (0 if kind == "plain" else 0.3)
                s, e = qtok[i].start, qtok[i + n - 1].end
                cands.append(Match(s, e, query[s:e], value, score, "lexicon"))
                break
    chosen: list[Match] = []
    for m in sorted(cands, key=lambda m: (-m.score, m.start)):
        if all(m.end <= c.start or m.start >= c.end for c in chosen):
            chosen.append(m)
    return sorted(chosen, key=lambda m: m.start)


@lru_cache(maxsize=None)
def _delete_index(field_name: str, keys: tuple[str, ...]) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for k in keys:
        if len(k) < 4:
            continue
        for d in {k} | {k[:i] + k[i + 1:] for i in range(len(k))}:
            idx.setdefault(d, []).append(k)
    return idx


def fuzzy_matches(fi: FieldInfo, query: str, qtok: list[Token], common: frozenset = frozenset()) -> list[Match]:
    """Surfaces one edit away (typos: «בעפול» -> עפולה, «סמס.ונג»), for 1-3 token windows of 4+ letters.
    A single frequent query word («מוצא», «מחפש») is never treated as a typo of a value."""
    idx = _delete_index(fi.name, tuple(sorted(fi.surfaces)))
    cands = []
    for i in range(len(qtok)):
        for n in (3, 2, 1):
            if i + n > len(qtok):
                continue
            for key, kind in _ngram_keys(qtok, i, n):
                if len(key) < 4 or (n == 1 and fold(qtok[i].text) in common):
                    continue
                pool = set(idx.get(key, []))
                for j in range(len(key)):
                    pool.update(idx.get(key[:j] + key[j + 1:], []))
                hits = [k for k in pool if k == key or edit1(key, k)]
                if hits:
                    k = max(hits, key=lambda h: sum(fi.surfaces[h].values()))
                    s, e = qtok[i].start, qtok[i + n - 1].end
                    cands.append(Match(s, e, query[s:e], _best_value(fi.surfaces[k]),
                                       1.5 * n - (0 if kind == "plain" else 0.3),
                                       "fuzzy"))
                    break
    chosen: list[Match] = []
    for m in sorted(cands, key=lambda m: (-m.score, m.start)):
        if all(m.end <= c.start or m.start >= c.end for c in chosen):
            chosen.append(m)
    return sorted(chosen, key=lambda m: m.start)


def combine_list(fi: FieldInfo, query: str, matches: list[Match]) -> Match | None:
    """One match -> it. Several distinct values joined by או / , / ו / '/' on a list-capable field -> a list."""
    if not matches:
        return None
    best = max(matches, key=lambda m: (m.score, -m.start))
    if len(matches) >= 2:
        # "X או Y" / "X/Y" makes a list on any enum field; list-capable fields also accept "," and a ו prefix
        loose = r"\s*(או|,|/|ו|או ש)?\s*" if fi.list_ok else r"\s*(או|/|או ש)\s*"
        group = [matches[0]]
        for m in matches[1:]:
            gap = query[group[-1].end:m.start]
            if re.fullmatch(loose, gap) or (fi.list_ok and re.fullmatch(r"\s*", gap) and m.text[:1] == "ו"):
                group.append(m)
            else:
                break
        values = []
        for m in group:
            for v in (m.value if isinstance(m.value, list) else [m.value]):
                if v not in values:
                    values.append(v)
        if len(values) >= 2:
            return Match(group[0].start, group[-1].end, query[group[0].start:group[-1].end], values,
                         sum(m.score for m in group), group[0].source)
    return best


_SIZE_UNIT = re.compile(r"\s*(?:ג['׳]?יג(?:ה|ות|י|ה)?|גיגה|גיגי|gb|GB|ג[\"״]ב|טרה|tb|TB)(?![א-תa-zA-Z])")


def unit_enum_match(fi: FieldInfo, query: str) -> Match | None:
    """Enum values that are sizes («64GB», «1TB»): a number (digits or words) followed by a size unit.
    Only values already known for the field are returned."""
    known = {v for counts in fi.surfaces.values() for v in map(json.loads, counts) if isinstance(v, str)}
    if not any(re.fullmatch(r"\d+(GB|TB)", v) for v in known):
        return None
    text = clean(query)
    for v in read_values(text):
        u = _SIZE_UNIT.match(text, v.end)
        if not u or not float(v.value).is_integer():
            continue
        tera = re.search(r"טרה|tb", u.group(0), re.I)
        cand = f"{int(v.value)}TB" if tera else f"{int(v.value)}GB"
        if cand in known:
            return Match(v.start, u.end(), query[v.start:u.end()], cand, 1.0, "numeric")
    return None


# ------------------------------------------------------------------ booleans
def bool_match(fi: FieldInfo, query: str, qtok: list[Token]) -> Match | None:
    ms = lexicon_matches(fi, query, qtok) or fuzzy_matches(fi, query, qtok)
    if not ms:
        return None
    m = max(ms, key=lambda m: m.score)
    i = next(k for k, t in enumerate(qtok) if t.start == m.start)
    neg = negated_before(qtok, i, window=2)
    s = qtok[i - 1].start if neg and i > 0 else m.start
    # the matched surface can start with the negation itself («בלי חיות» is a train surface) or glue it on
    # («בלימיזוג», «בליממד»); bool fields have no value word starting with בלי/ללא
    first = qtok[i].text
    inner = any(v in NEGATIONS for v in prefix_variants(first)) or (len(first) >= 6 and first.startswith(("בלי", "ללא")))
    return Match(s, m.end, query[s:m.end], not (neg or inner), m.score, m.source)


# ------------------------------------------------------------------ numbers
@dataclass
class NumCand:
    start: int
    end: int
    text: str
    n_values: int


def numeric_candidates(query: str, qtok: list[Token], units: frozenset = frozenset(),
                       domain: str | None = None) -> list[NumCand]:
    """Number mentions extended with bound/unit words; two mentions joined by a connector also form a range.
    Outside vehicles, «יד שנייה» / «יד 2» names the site (second-hand), not a number."""
    text = clean(query)
    vals = read_values(text)
    blocked = [(m.start(), m.end()) for m in NUMBERED_PLACES.finditer(text)]
    if domain != "רכב":
        blocked += [(m.start(), m.end()) for m in SECOND_HAND.finditer(text)]
    vals = [v for v in vals if not any(v.start >= s and v.end <= e for s, e in blocked)]
    if not vals:
        return []

    def tok_idx(pos_start, pos_end):
        ids = [i for i, t in enumerate(qtok) if t.end > pos_start and t.start < pos_end]
        return (ids[0], ids[-1]) if ids else (None, None)

    def extend(a, b):
        # bound words before the number, possibly separated from it by unit words: "לכל היותר יד אחת",
        # "לא מעל קומה 9"; unit words are only kept when a bound word lies beyond them
        k, units_seen = a, 0
        while k > 0 and b - k < 7:
            w = prefix_variants(qtok[k - 1].text)
            if any(fold(x) in BOUND_LEFT for x in w):
                k -= 1
                a = k
            elif _stem(qtok[k - 1].text) in units and units_seen < 2:
                k -= 1
                units_seen += 1
            else:
                break
        if a > 0 and fold(qtok[a - 1].text) in {"מ", "ל"}:
            a -= 1
        steps = 0  # unit words right after the number ("חדרים", "ש\"ח", "דונם"), then a trailing bound word;
        # a left-bound word ("עד", "בין", "מ-") belongs to the NEXT number: "4 חדרים עד 2 מיליון"
        while b + 1 < len(qtok) and steps < 2 and _stem(qtok[b + 1].text) in units \
                and not any(fold(x) in BOUND_LEFT for x in prefix_variants(qtok[b + 1].text)):
            b += 1
            steps += 1
        while b + 1 < len(qtok) and fold(qtok[b + 1].text) in BOUND_RIGHT:
            b += 1
        return a, b

    spans = []
    for k, v in enumerate(vals):
        a, b = tok_idx(v.start, v.end)
        if a is None:
            continue
        spans.append((a, b, 1))
        if k + 1 < len(vals) and CONNECTOR.match(text[v.end:vals[k + 1].start] or "") \
                and len(text[v.end:vals[k + 1].start]) <= 8:
            a2, b2 = tok_idx(vals[k + 1].start, vals[k + 1].end)
            if a2 is not None:
                spans.append((a, b2, 2))
        elif k + 1 < len(vals) and (
                (re.search(r"(^|\s)(ו?עד|ל|או)\s", text[v.end:vals[k + 1].start])
                 and len(text[v.end:vals[k + 1].start].split()) <= 3)
                # "בין קומה 0 לקומה 5": one ל-prefixed word between the values, after a «בין»
                or (re.fullmatch(r"\s*ל[א-ת]+\s*", text[v.end:vals[k + 1].start])
                    and any(fold(t.text) == "בינ" for t in qtok[max(0, a - 3):a]))):
            a2, b2 = tok_idx(vals[k + 1].start, vals[k + 1].end)
            if a2 is not None:
                spans.append((a, b2, 2))
    out, seen = [], set()
    for a, b, n in spans:
        a, b = extend(a, b)
        s, e = qtok[a].start, qtok[b].end
        if (s, e) not in seen:
            seen.add((s, e))
            out.append(NumCand(s, e, query[s:e], n))
    return out


def numeric_score(fi: FieldInfo, cand: NumCand, query: str, qtok: list[Token], totals: dict[str, int],
                  k_fields: int) -> tuple[float, str | None, object]:
    """Cue evidence around the candidate for this field + plausibility of the value -> (score, op, value)."""
    op, value = parse_numeric(cand.text, fi.name)
    if value is None:
        return -1e9, None, None
    ids = [i for i, t in enumerate(qtok) if t.end > cand.start and t.start < cand.end]
    window = range(max(0, ids[0] - 2), min(len(qtok), ids[-1] + 2)) if ids else []
    alpha, score = 0.2, 0.0
    for i in window:
        w = _stem(qtok[i].text)
        cf, tot = fi.cues.get(w, 0), totals.get(w, 0)
        if tot:
            score += math.log((cf + alpha) / (tot + alpha * k_fields)) - math.log(1.0 / k_fields)
    xs = [x for x in (value.values() if isinstance(value, dict) else [value]) if isinstance(x, (int, float))]
    if xs and min(xs) > 0:
        z = (math.log10(max(xs)) - fi.log_mu) / fi.log_sigma
        score -= 0.5 * min(z * z, 25)
    elif xs and min(xs) == 0:
        score -= 1.0
    if cand.n_values == 2 and op == "range" and xs and min(xs) > 0 and max(xs) / min(xs) > 1000:
        score -= 3.0  # «4 חדרים עד 2 מיליון»: two numbers of different fields, not one range
    elif cand.n_values == 2 and op == "range":
        score += 2.0  # the whole range beats either of its halves
    elif op in ("min", "max"):
        score += 0.5
    if _CORRECTION.search(query[:cand.start]):
        score += 1.5  # "2001 סליחה 2010": the value after a self-correction wins
    return score, op, value


_CORRECTION = re.compile(r"בעצם|סליחה|רגע|כלומר|טעות|(?:הת)?כוונתי")


def _stem(tok: str) -> str:
    """Folded token with one-letter prefixes removed (kept if that would leave 2 letters or fewer)."""
    core = prefix_variants(tok)[-1]
    return fold(core) if len(core) > 2 else fold(tok)


# ------------------------------------------------------------------ dates
_MONTHS = {"ינואר": 1, "פברואר": 2, "מרץ": 3, "מרס": 3, "אפריל": 4, "מאי": 5, "יוני": 6, "יולי": 7, "אוגוסט": 8,
           "ספטמבר": 9, "אוקטובר": 10, "נובמבר": 11, "דצמבר": 12}
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_PATTERNS = [
    (re.compile(r"(?<!\d)(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{4}|\d{2})(?!\d)"), "dmy"),
    (re.compile(r"(?<![א-ת])[ובל]?(" + _MON + r")(?![א-ת])"), "month"),
    (re.compile(r"(?<![\d./-])(\d{1,2})\s*[./-]\s*(\d{4})(?!\d)"), "my"),
    (re.compile(r"(?<![א-ת])(?:ל|ב)?(מיידי|מיידית|מידי|מידית|מייד)(?![א-ת])"), "now"),
]


def _month_date(text: str, m) -> tuple[str | None, int, int]:
    """A month name anchors the date; the year (digits or words) follows it, an optional day precedes it:
    "22 במאי 2027", "בשמונה במרץ 2027", "אוקטובר אלפיים עשרים ושבע", "מרץ 2027"."""
    month = _MONTHS[m.group(1)]
    after = text[m.end():m.end() + 40]
    years = [v for v in read_values(after) if 1990 <= v.value <= 2100]
    if not years:
        return None, m.start(), m.end()
    y = years[0]
    end = m.end() + y.end
    before = text[max(0, m.start() - 15):m.start()]
    days = [v for v in read_values(before) if 1 <= v.value <= 31 and float(v.value).is_integer()]
    if days and not before[days[-1].end:].strip(" ב"):
        start = m.start() - len(before) + days[-1].start
        return f"{int(days[-1].value):02d}-{month:02d}-{int(y.value)}", start, end
    return f"{month:02d}-{int(y.value)}", m.start(), end


def _year(y: str) -> int:
    return int(y) + 2000 if len(y) == 2 else int(y)


def date_matches(query: str) -> list[Match]:
    text, out = clean(query), []
    for rx, kind in _DATE_PATTERNS:
        for m in rx.finditer(text):
            if any(m.start() < o.end and m.end() > o.start for o in out):
                continue
            s, e = m.start(), m.end()
            try:
                if kind == "dmy":
                    d, mo, y = int(m.group(1)), int(m.group(2)), _year(m.group(3))
                    val = f"{d:02d}-{mo:02d}-{y}" if 1 <= mo <= 12 and 1 <= d <= 31 else None
                elif kind == "month":
                    val, s, e = _month_date(text, m)
                elif kind == "my":
                    mo = int(m.group(1))
                    val = f"{mo:02d}-{m.group(2)}" if 1 <= mo <= 12 else None
                else:
                    val = "מיידי"
            except (ValueError, KeyError):
                val = None
            if val:
                out.append(Match(s, e, query[s:e], val, 1.0, "date"))
    return sorted(out, key=lambda m: m.start)


DATE_CUES = {"תאריך_כניסה": ("כניסה", "להיכנס", "פינוי", "כניס"), "טסט_עד": ("טסט", "טסס", "רישוי")}


def pick_date(field_name: str, query: str, dates: list[Match]) -> Match | None:
    if not dates:
        return None
    cues = DATE_CUES.get(field_name.split("/", 1)[-1], ())
    pos = [m.start() for c in cues for m in re.finditer(c, query)]
    if pos:
        return min(dates, key=lambda d: min(abs(d.start - p) for p in pos))
    return dates[0]


# ------------------------------------------------------------------ open-vocabulary text by cue words
_STOP = r"(?:עם|ליד|קרוב|עד|בין|ב|של|או|שיהיה|כולל|מ|ו|ומעלה|לפחות|ממש|בבקשה|חדרים|קומה|מחיר)"
_TAIL = r"(?:\s+(?!" + _STOP + r"(?:\s|$))[^\s,.!?\d]+)?"
# typical first words of Israeli neighbourhood names, used when no «שכונת» cue is written («ברמת פולג»)
_HOOD_HEADS = r"(?:רמת|רמות|גבעת|גני|נווה|נוה|נאות|מעוז|קרית|קריית|כפר|פסגת|שיכון|תל|נחלת|רסקו|מורשה|הדר|שערי)"
_CUE_PATTERNS = {  # tried in order: an explicit cue word first, a typical name head only as a fallback
    "רחוב": [re.compile(r"(?<![א-ת])ו?ב?רח(?:וב|')\s+(?P<v>[^\s,.!?]+" + _TAIL + ")")],
    "שכונה": [re.compile(r"(?<![א-ת])ו?ב?שכונ(?:ת|ה)\s+(?P<v>[^\s,.!?]+" + _TAIL + ")"),
              re.compile(r"(?<![א-ת])ו?ב(?P<h>" + _HOOD_HEADS + r"[\s-]+[^\s,.!?\d]+)")],
    "מותג": [re.compile(r"(?<![א-ת])(?:מותג|של|מבית|מתוצרת)\s+(?P<v>[^\s,.!?]+)")],
}


def cue_text_match(field_name: str, query: str, common: frozenset = frozenset()) -> Match | None:
    """'ברחוב X', 'בשכונת X Y', 'של X': the words after the cue, stopping at a common query word."""
    m = None
    for rx in _CUE_PATTERNS.get(field_name.split("/", 1)[-1], []):
        m = rx.search(clean(query))
        if m:
            break
    if not m:
        return None
    group = "v" if m.groupdict().get("v") else "h"
    raw = m.group(group)
    words = raw.split()
    if group == "v" and len(words) > 1 and fold(words[1]) in common:
        words = words[:1]
    v = " ".join(words).strip(" \"'")
    end = m.start(group) + len(" ".join(words))
    return Match(m.start(), end, query[m.start():end], v, 0.5 if group == "v" else 0.3, "cue")
