"""Label schema v2 ("point fields carry bounds"): deterministic load-time normalization.

Decision (docs/taxonomy_point_range_inconsistency.md, option A + year fold): the 7 `טווח_*` range filters and the
query-level vehicle `שנות_ייצור` are renamed to their point field. A bounded value keeps its value object
({"min": a} / {"max": b} / {"min": a, "max": b}), so the operator is preserved by the value and by the span `op`.

The data files are NEVER modified: callers apply normalize_row() to each row when loading
(see training/shared/dataset.py). Pure functions, no I/O.

Not renamed on purpose: `רכב/טווח_חשמלי_ק״מ` is an EV's driving range (a point field whose name happens to start
with `טווח_`), and the taxonomy's per-manufacturer `יצרנים.<מותג>.שנות_ייצור` catalogue spans stay as metadata.
"""
from __future__ import annotations

import copy
import hashlib
import json

SCHEMA = "v2-point-fields"

RENAMES: dict[str, dict[str, str]] = {
    "נדל״ן": {"טווח_מחיר": "מחיר", "טווח_קומה": "קומה", "טווח_חדרים": "מס׳_חדרים", "טווח_מ״ר": "מ״ר_בנוי"},
    "רכב": {"טווח_מחיר": "מחיר", "טווח_שנה": "שנה", "טווח_ק״מ": "ק״מ", "שנות_ייצור": "שנה"},
}
NOT_RENAMED = frozenset({"רכב/טווח_חשמלי_ק״מ"})
RULES_SHA = hashlib.sha256(json.dumps({"schema": SCHEMA, "renames": RENAMES, "keep": sorted(NOT_RENAMED)},
                                      ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]

BOUNDED_OPS = {"min", "max", "range"}


class NormalizationError(ValueError):
    pass


def _shape_ok(value, op: str | None) -> bool:
    if op is None:
        return True
    if op == "exact":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if op in BOUNDED_OPS:
        return isinstance(value, dict) and bool(value) and set(value) <= {"min", "max"}
    return True


def _rename_keys(d: dict, mapping: dict[str, str], where: str, rid: str) -> tuple[dict, list[list[str]]]:
    out, renamed = {}, []
    for k, v in d.items():
        nk = mapping.get(k, k)
        if nk in out or (nk != k and nk in d):
            raise NormalizationError(f"{rid}: renaming {where} '{k}' -> '{nk}' collides with an existing field")
        out[nk] = v
        if nk != k:
            renamed.append([k, nk])
    return out, renamed


def normalize_row(row: dict) -> dict:
    """Return a normalized deep copy of a dataset row (input untouched). Idempotent."""
    new = copy.deepcopy(row)
    rid = str(row.get("id"))
    mapping = RENAMES.get(row.get("domain") or "", {})
    renamed: list[list[str]] = []
    warnings: list[str] = []
    expected = new.get("expected") or {}
    if mapping and isinstance(expected.get("params"), dict):
        expected["params"], renamed = _rename_keys(expected["params"], mapping, "param", rid)
    if mapping and isinstance(new.get("spans"), dict):
        new["spans"], span_renamed = _rename_keys(new["spans"], mapping, "span", rid)
        if sorted(map(tuple, span_renamed)) != sorted(map(tuple, renamed)):
            warnings.append("params and spans renamed different fields")
    for src, dst in renamed:
        spans = (new.get("spans") or {}).get(dst) or [{}]
        if not _shape_ok(expected["params"][dst], spans[0].get("op")):
            warnings.append(f"{dst}: value shape does not match span op {spans[0].get('op')!r}")
    prev = row.get("_normalization") or {}
    new["_normalization"] = {"schema": SCHEMA, "rules_sha": RULES_SHA,
                             "renamed": (prev.get("renamed") or []) + renamed,
                             "warnings": (prev.get("warnings") or []) + warnings}
    return new


def normalize_field(field: str) -> str:
    """'רכב/טווח_מחיר' -> 'רכב/מחיר' (domain-qualified field names)."""
    dom, _, name = field.partition("/")
    return f"{dom}/{RENAMES.get(dom, {}).get(name, name)}"


def normalize_fields(fields: list[str]) -> list[str]:
    out = []
    for f in fields:
        n = normalize_field(f)
        if n not in out:
            out.append(n)
    return out


def normalize_taxonomy(taxonomy: dict) -> dict:
    """View of the taxonomy under schema v2 (deep copy; the file is not touched).

    Drops the `טווח_*` entries that were renamed from each domain's `פילטרים_נוספים` (the section is removed if it
    becomes empty) and records that numeric point fields accept {"min", "max"}.
    """
    tax = copy.deepcopy(taxonomy)
    cats = tax.get("קטגוריות") or {}
    dropped = {}
    for dom, mapping in RENAMES.items():
        filters = (cats.get(dom) or {}).get("פילטרים_נוספים")
        if not isinstance(filters, dict):
            continue
        gone = [k for k in list(filters) if k in mapping]
        for k in gone:
            del filters[k]
        if not filters:
            del cats[dom]["פילטרים_נוספים"]
        dropped[dom] = gone
    tax["_normalization"] = {
        "schema": SCHEMA, "rules_sha": RULES_SHA, "dropped_range_filters": dropped,
        "note": "numeric point fields hold a number (exact) or {'min': a} / {'max': b} / {'min': a, 'max': b}",
        "kept": sorted(NOT_RENAMED),
    }
    return tax
