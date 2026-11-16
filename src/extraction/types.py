"""Result types and type-aware value equality."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

SOURCES = ("lexicon", "fuzzy", "numeric", "date", "cue", "tagger", "taxonomy")


@dataclass
class Extraction:
    field: str                       # domain-qualified, e.g. "נדל״ן/מחיר"
    value: object = None             # None = not found
    operator: str | None = None      # exact | min | max | range for numeric fields
    start: int | None = None         # span offsets into the query
    end: int | None = None
    text: str | None = None
    source: str | None = None        # which component produced the value (see SOURCES)
    score: float = 0.0
    reason: str | None = None        # why nothing was found

    @property
    def found(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict:
        return asdict(self)


def _norm_str(s: str) -> str:
    s = s.replace("״", '"').replace("׳", "'").replace("–", "-").replace("־", "-")
    return re.sub(r"\s+", " ", s).strip()


def values_equal(a, b) -> bool:
    """Type-aware equality used by the evaluation: numbers exact (int/float), {min,max} by bound, lists as
    sets, strings after quote/space normalization, booleans exact."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(values_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) or isinstance(b, list):
        la = a if isinstance(a, list) else [a]
        lb = b if isinstance(b, list) else [b]
        return sorted(map(lambda x: _norm_str(str(x)), la)) == sorted(map(lambda x: _norm_str(str(x)), lb))
    if isinstance(a, str) and isinstance(b, str):
        return _norm_str(a) == _norm_str(b)
    return False
