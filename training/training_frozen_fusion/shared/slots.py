"""Label space: 77 domain-qualified fields -> 70 slots (7 twin pairs: 4 in נדל״ן, 3 in רכב) (each point/range twin pair becomes one slot).

The twins never co-occur in a query (0 of 6,171 train rows), so the mapping is lossless for *presence detection*.
It is NOT lossless for JSON reconstruction: the output still needs the taxonomy field name, which
taxonomy_rules.resolve() picks once the field's operator is known.
"""
from __future__ import annotations

TWINS = {  # range field -> point field (within the same domain)
    "טווח_מחיר": "מחיר",
    "טווח_קומה": "קומה",
    "טווח_ק״מ": "ק״מ",
    "טווח_חדרים": "מס׳_חדרים",
    "טווח_מ״ר": "מ״ר_בנוי",
    "טווח_שנה": "שנה",
}
POINT_TO_RANGE = {v: k for k, v in TWINS.items()}
SEP = "⇄"


def to_slot(field: str, vocab: set[str] | None = None) -> str:
    """'רכב/טווח_מחיר' and 'רכב/מחיר' -> 'רכב/מחיר⇄טווח_מחיר' (only if both twins exist in that domain)."""
    dom, name = field.split("/", 1)
    point = TWINS.get(name, name)
    rng = POINT_TO_RANGE.get(point)
    if rng is None:
        return field
    if vocab is not None and not ({f"{dom}/{point}", f"{dom}/{rng}"} <= vocab):
        return field
    return f"{dom}/{point}{SEP}{rng}"


def is_twin_slot(slot: str) -> bool:
    return SEP in slot


def slot_names(slot: str) -> tuple[str, str]:
    """Twin slot -> (point field, range field), both domain-qualified."""
    dom, pair = slot.split("/", 1)
    point, rng = pair.split(SEP)
    return f"{dom}/{point}", f"{dom}/{rng}"


def slot_vocab(vocab77: list[str]) -> list[str]:
    v = set(vocab77)
    return sorted({to_slot(f, v) for f in vocab77})


def fields_to_slots(fields: list[str], vocab77: list[str]) -> list[str]:
    v = set(vocab77)
    out = []
    for f in fields:
        s = to_slot(f, v)
        if s not in out:
            out.append(s)
    return out
