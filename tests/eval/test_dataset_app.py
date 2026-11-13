"""Pure-logic tests for the ground-truth dataset editor (no Streamlit)."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
APP_DIR = REPO / "eval" / "dataset" / "app"
sys.path.insert(0, str(APP_DIR))

import store  # noqa: E402
from inventory_index import load_index, normalize_hebrew  # noqa: E402

INVENTORY = REPO / "eval" / "dataset" / "inventory" / "field_inventory.json"
DIMENSIONS = REPO / "eval" / "dataset" / "dimensions" / "dimensions_matrix.json"


@pytest.fixture(scope="module")
def index():
    return load_index(INVENTORY, DIMENSIONS)


def rec(category, params, metadata=None, query="שאילתה", rid="gt_0001"):
    return {"id": rid, "query": query, "expected": {"category": category, "params": params},
            "metadata": metadata or {}, "notes": ""}


def test_domain_quote_normalized(index):
    assert normalize_hebrew('נדל"ן') == "נדל״ן"
    assert index.dimension("domain").values == index.domains


def test_hierarchy_keys(index):
    assert index.hierarchy_keys["רכב"] == {"יצרנים", "דגמים"}
    assert index.hierarchy_keys["יד_שנייה"] == {"סקטורים", "תתי_קטגוריות"}
    assert index.hierarchy_keys["נדל״ן"] == set()


def test_available_fields_follow_conditions(index):
    none = index.available_fields("רכב", {})
    assert "שנות_ייצור" not in none and "תתי_דגמים" not in none and "שנה" in none
    toyota = index.available_fields("רכב", {"יצרנים": "טויוטה"})
    assert "שנות_ייצור" in toyota and "תתי_דגמים" not in toyota
    corolla = index.available_fields("רכב", {"יצרנים": "טויוטה", "דגמים": "קורולה"})
    assert corolla["תתי_דגמים"]["possible_values"] == ["LE", "SE", "Hybrid", "Sun"]


def test_most_specific_field_wins(index):
    general = index.available_fields("יד_שנייה", {})
    assert general["צבע"]["type"] == "string"
    phones = index.available_fields("יד_שנייה", {"סקטורים": "אלקטרוניקה", "תתי_קטגוריות": "טלפונים_סלולריים"})
    assert phones["צבע"]["type"] == "enum"
    assert "מותגים" in phones


def test_validate_ok(index):
    r = rec("רכב", {"יצרנים": "טויוטה", "דגמים": "קורולה", "שנה": {"min": 2018, "max": 2021},
                    "מחיר": {"max": 70000}, "צבע": "לבן"},
            {"domain": ["רכב"], "operator": ["exact", "range", "max"]})
    errors, warnings = store.validate(r, index)
    assert errors == [] and warnings == []


@pytest.mark.parametrize("category,params,metadata,needle", [
    ("רכב", {"תתי_דגמים": "LE"}, {}, "תתי_דגמים"),                 # level not reachable
    ("רכב", {"צבע": "ורוד"}, {}, "not in possible values"),
    ("נדל״ן", {"לא_קיים": 1}, {}, "not a field"),
    ("נדל״ן", {"מס׳_חדרים": {"min": 5, "max": 2}}, {}, "min > max"),
    ("נדל״ן", {"מעלית": "כן"}, {}, "true/false"),
    (None, {"מחיר": 5}, {}, "params must be empty"),
    ("רכב", {}, {"language": ["klingon"]}, "metadata.language"),
    ("רכב", {}, {"unknown_dim": ["x"]}, "unknown dimension"),
])
def test_validate_errors(index, category, params, metadata, needle):
    errors, _ = store.validate(rec(category, params, metadata), index)
    assert any(needle in e for e in errors), errors


def test_out_of_range_is_warning(index):
    errors, warnings = store.validate(rec("נדל״ן", {"מס׳_חדרים": 50}, {"domain": ["נדל״ן"]}), index)
    assert errors == [] and any("outside inventory range" in w for w in warnings)


def test_null_category_allowed(index):
    errors, _ = store.validate(rec(None, {}, {"complexity": ["unsupported"], "field_count": ["zero"]}), index)
    assert errors == []


def test_save_roundtrip_sorted_and_ids(tmp_path, index):
    path = tmp_path / "data" / "gt.jsonl"
    records = [rec("רכב", {}, rid="gt_0010"), rec(None, {}, rid="gt_0002", query="אחרת")]
    store.save(path, records, index)
    loaded = store.load(path)
    assert [r["id"] for r in loaded] == ["gt_0002", "gt_0010"]
    assert list(loaded[0]) == ["id", "query", "expected", "metadata", "notes"]
    assert list(loaded[0]["metadata"]) == [d.name for d in index.dimensions]
    assert "רכב" in path.read_text(encoding="utf-8")  # ensure_ascii=False
    assert store.next_id(loaded) == "gt_0011"
    assert not list(path.parent.glob("*.tmp"))
