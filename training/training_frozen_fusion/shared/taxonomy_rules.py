"""Deterministic taxonomy post-processing, kept separate from language understanding.

Applied in this iteration (they only need field *presence*):
- resolve(slot, local_span, operator): picks the taxonomy field name of a point/range twin slot. A known operator
  follows the taxonomy (point fields live under מאפיינים_כלליים as numbers, טווח_* under פילטרים_נוספים as
  {min, max}): bounded -> טווח_*, exact -> point field. Without an operator it returns a fixed default name per
  slot (the train-majority name). It never looks at the whole query.
- close_hierarchy(fields): adds parents implied by children. Measured as exceptionless in train and val:
  תתי_קטגוריות -> סקטורים (1718/1718), דגמים -> יצרנים (1165/1165), תתי_דגמים -> דגמים (128/128).

Implemented but NOT applied yet (they need field *values*, i.e. the value extractor):
- storage_field_for(sub_category): laptops -> אחסון_GB, phones -> נפח_אחסון (from the taxonomy).
- valid_model(manufacturer, model): manufacturer/model combinations listed in the taxonomy.
- sector_for(sub_category): parent sector of a sub-category value.
"""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache

from .operators import is_bounded
from .paths import TAXONOMY
from .slots import is_twin_slot, slot_names

HIERARCHY = {  # child field -> parent field (same domain)
    "יד_שנייה/תתי_קטגוריות": "יד_שנייה/סקטורים",
    "רכב/דגמים": "רכב/יצרנים",
    "רכב/תתי_דגמים": "רכב/דגמים",
}


class Resolver:
    def __init__(self, default_names: dict[str, str]):
        self.default_names = default_names  # twin slot -> field name used when no operator is known

    @classmethod
    def from_train(cls, train_field_lists: list[list[str]], slots: list[str]) -> "Resolver":
        counts = Counter(f for fields in train_field_lists for f in fields)
        defaults = {}
        for s in slots:
            if is_twin_slot(s):
                point, rng = slot_names(s)
                defaults[s] = point if counts[point] >= counts[rng] else rng
        return cls(defaults)

    def resolve(self, slot: str, local_span: str | None = None, operator: str | None = None) -> str:
        if not is_twin_slot(slot):
            return slot
        point, rng = slot_names(slot)
        if operator is not None:
            return rng if is_bounded(operator) else point
        return self.default_names[slot]


def close_hierarchy(fields: list[str]) -> list[str]:
    out = list(fields)
    changed = True
    while changed:
        changed = False
        for child, parent in HIERARCHY.items():
            if child in out and parent not in out:
                out.append(parent)
                changed = True
    return out


@lru_cache(maxsize=1)
def _taxonomy() -> dict:
    return json.loads(TAXONOMY.read_text(encoding="utf-8"))["קטגוריות"]


def sector_for(sub_category: str) -> str | None:
    for sector, body in _taxonomy()["יד_שנייה"]["סקטורים"].items():
        if sub_category in (body.get("תתי_קטגוריות") or {}):
            return sector
    return None


def storage_field_for(sub_category: str) -> str | None:
    for body in _taxonomy()["יד_שנייה"]["סקטורים"].values():
        attrs = (body.get("תתי_קטגוריות") or {}).get(sub_category)
        if attrs is not None:
            for name in ("אחסון_GB", "נפח_אחסון"):
                if name in attrs:
                    return f"יד_שנייה/{name}"
    return None


def valid_model(manufacturer: str, model: str) -> bool | None:
    """True/False if the taxonomy lists the manufacturer, None if it does not know it."""
    m = _taxonomy()["רכב"]["יצרנים"].get(manufacturer)
    if m is None:
        return None
    return model in (m.get("דגמים") or [])
