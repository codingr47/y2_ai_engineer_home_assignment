"""Single loading path for the split1 data and the taxonomy, with a label schema.

    schema="v2"  (default) -> rows pass through normalize.normalize_row (point fields carry bounds)
    schema="raw"           -> rows exactly as stored (Experiment 1 baseline, data documentation)

The files are only ever read. val_calib / val_eval are the committed experiment_1 split
(experiment_1/artifacts/labels/val_split.json is regenerable; it is rebuilt by experiment_1/preprocess.py).
"""
from __future__ import annotations

import json
from functools import lru_cache

from . import normalize
from .paths import DATA_DIR, EXP1, TAXONOMY

SCHEMAS = ("raw", "v2")
_FILE = {"train": "train.jsonl", "val": "val.jsonl", "val_calib": "val.jsonl", "val_eval": "val.jsonl",
         "test": "test.jsonl"}


@lru_cache(maxsize=None)
def _read(fname: str) -> tuple[dict, ...]:
    with open(DATA_DIR / fname, encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f if line.strip())


@lru_cache(maxsize=None)
def _val_split() -> dict[str, list[str]]:
    return json.loads((EXP1 / "artifacts" / "labels" / "val_split.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _rows(split: str, schema: str) -> tuple[dict, ...]:
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema {schema!r}; expected one of {SCHEMAS}")
    rows = _read(_FILE[split])
    if split in ("val_calib", "val_eval"):
        by_id = {r["id"]: r for r in rows}
        rows = tuple(by_id[i] for i in _val_split()[split])
    return tuple(normalize.normalize_row(r) for r in rows) if schema == "v2" else rows


def load_rows(split: str, schema: str = "v2") -> list[dict]:
    """Rows of a split in file order (val_calib / val_eval in committed split order). Returns copies-safe tuples'
    contents; callers must not mutate them."""
    return list(_rows(split, schema))


def rows_by_id(split: str, schema: str = "v2") -> dict[str, dict]:
    return {r["id"]: r for r in _rows(split, schema)}


def field_labels(row: dict) -> list[str]:
    d = row.get("domain")
    if d is None:
        return []
    return [f"{d}/{k}" for k in ((row.get("expected") or {}).get("params") or {})]


@lru_cache(maxsize=None)
def field_vocab(schema: str = "v2") -> tuple[str, ...]:
    vocab = sorted({f for r in _rows("train", schema) for f in field_labels(r)})
    if schema == "v2":
        stray = [f for f in vocab if f.split("/", 1)[1].startswith("טווח_") and f not in normalize.NOT_RENAMED]
        assert not stray, f"range-filter fields survived normalization: {stray}"
    return tuple(vocab)


def load_taxonomy(schema: str = "v2") -> dict:
    tax = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    return normalize.normalize_taxonomy(tax) if schema == "v2" else tax
