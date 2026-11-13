"""JSONL storage + validation for ground-truth records.

Record shape (one JSON object per line):
    {"id": "gt_0001",
     "query": "...",
     "expected": {"category": "רכב" | null, "params": {...}},
     "metadata": {"<dimension>": ["<value>", ...], ...},
     "notes": ""}
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from inventory_index import InventoryIndex, is_number

ID_PREFIX = "gt_"
_ID_RE = re.compile(rf"^{ID_PREFIX}(\d+)$")

NUMERIC_TYPES = {"number", "range"}
TEXT_TYPES = {"string", "date", "date_relative_or_exact"}


# --- storage ----------------------------------------------------------------
def load(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON: {exc}") from exc
    return records


def canonical(record: dict, index: InventoryIndex | None = None) -> dict:
    """Fixed key order; metadata ordered as in the dimensions matrix."""
    meta = record.get("metadata") or {}
    if index is not None and index.dimensions:
        order = [d.name for d in index.dimensions]
        meta = {k: list(meta.get(k, [])) for k in order} | {k: v for k, v in meta.items() if k not in order}
    exp = record.get("expected") or {}
    return {
        "id": record["id"],
        "query": record.get("query", ""),
        "expected": {"category": exp.get("category"), "params": exp.get("params") or {}},
        "metadata": meta,
        "notes": record.get("notes", ""),
    }


def _sort_key(rec: dict):
    m = _ID_RE.match(rec.get("id", ""))
    return (0, int(m.group(1)), "") if m else (1, 0, rec.get("id", ""))


def save(path: Path, records: list[dict], index: InventoryIndex | None = None) -> None:
    """Atomically rewrite the whole file, sorted by id."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(canonical(r, index), ensure_ascii=False) for r in sorted(records, key=_sort_key)]
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def next_id(records: list[dict]) -> str:
    nums = [int(m.group(1)) for r in records if (m := _ID_RE.match(r.get("id", "")))]
    return f"{ID_PREFIX}{(max(nums) + 1 if nums else 1):04d}"


def upsert(records: list[dict], record: dict) -> list[dict]:
    out = [r for r in records if r.get("id") != record["id"]]
    out.append(record)
    return out


# --- validation -------------------------------------------------------------
def _check_value(name: str, entry: dict, value) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    t = entry["type"]

    if t.startswith("enum"):
        items = value if isinstance(value, list) else [value]
        if not items:
            errors.append(f"{name}: empty value")
        allowed = entry.get("possible_values") or []
        bad = [v for v in items if v not in allowed]
        if bad:
            errors.append(f"{name}: {bad} not in possible values")

    elif t in NUMERIC_TYPES:
        if is_number(value):
            nums = [value]
        elif isinstance(value, dict) and value and set(value) <= {"min", "max"} \
                and all(is_number(v) for v in value.values()):
            nums = list(value.values())
            if "min" in value and "max" in value and value["min"] > value["max"]:
                errors.append(f"{name}: min > max")
        else:
            errors.append(f"{name}: expected a number or {{min,max}}, got {value!r}")
            nums = []
        rng = entry.get("range")
        if rng:
            lo, hi = rng.get("min"), rng.get("max")
            for v in nums:
                if (lo is not None and v < lo) or (hi is not None and v > hi):
                    warnings.append(f"{name}: {v} outside inventory range [{lo}, {hi}]")

    elif t == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{name}: expected true/false, got {value!r}")

    elif t in TEXT_TYPES:
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{name}: expected non-empty text")

    else:
        warnings.append(f"{name}: unknown field type {t!r}, value not checked")
    return errors, warnings


def validate(record: dict, index: InventoryIndex, records: list[dict] | None = None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block saving."""
    errors: list[str] = []
    warnings: list[str] = []

    query = (record.get("query") or "").strip()
    if not query:
        errors.append("query is empty")

    exp = record.get("expected") or {}
    category = exp.get("category")
    params = exp.get("params") or {}

    if category is None:
        if params:
            errors.append("params must be empty when category is null")
    elif category not in index.domains:
        errors.append(f"unknown category {category!r}")
    else:
        picks = index.picks_from_params(category, params)
        fields = index.available_fields(category, picks)
        for name, value in params.items():
            if name in index.hierarchy_keys.get(category, set()):
                entry = index.hierarchy_entry(category, name, picks)
                if entry is None:
                    errors.append(f"{name}: hierarchy level not reachable with picks {picks}")
                    continue
                if not isinstance(value, str):
                    errors.append(f"{name}: hierarchy value must be a single string")
                    continue
            elif name in fields:
                entry = fields[name]
            else:
                errors.append(f"{name}: not a field of {category} for picks {picks}")
                continue
            e, w = _check_value(name, entry, value)
            errors += e
            warnings += w

    meta = record.get("metadata") or {}
    for dim_name, values in meta.items():
        dim = index.dimension(dim_name)
        if dim is None:
            errors.append(f"metadata: unknown dimension {dim_name!r}")
            continue
        if not isinstance(values, list):
            errors.append(f"metadata.{dim_name}: must be a list")
            continue
        bad = [v for v in values if v not in dim.values]
        if bad:
            errors.append(f"metadata.{dim_name}: {bad} not in {dim.values}")
    if index.dimensions and not any(meta.get(d.name) for d in index.dimensions):
        warnings.append("metadata: no dimension values selected")
    dom = meta.get("domain") or []
    if category and dom and category not in dom:
        warnings.append(f"metadata.domain {dom} does not include category {category!r}")

    if records and query:
        dup = [r["id"] for r in records if r.get("id") != record.get("id")
               and (r.get("query") or "").strip() == query]
        if dup:
            warnings.append(f"duplicate query text (also in {', '.join(dup)})")

    return errors, warnings
