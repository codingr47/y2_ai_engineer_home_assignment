import copy
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/training_frozen_fusion"))

from shared import normalize  # noqa: E402
from shared.normalize import NormalizationError, normalize_field, normalize_fields, normalize_row  # noqa: E402

DATA = ROOT / "training/training_frozen_fusion" / "data" / "split1"


def car_row(params):
    spans = {k: [{"start": 0, "end": 1, "text": "x", "op": ("exact" if not isinstance(v, dict) else
                                                          "range" if len(v) == 2 else next(iter(v))),
                  "value": v, "origin": "explicit"}] for k, v in params.items()}
    return {"id": "t1", "query": "q", "domain": "רכב", "expected": {"category": "רכב", "params": dict(params)},
            "spans": spans}


def test_range_filter_renamed_in_params_and_spans_with_value_and_order_kept():
    row = car_row({"יצרנים": "טויוטה", "טווח_מחיר": {"max": 70000}, "צבע": "לבן"})
    out = normalize_row(row)
    assert list(out["expected"]["params"]) == ["יצרנים", "מחיר", "צבע"]
    assert out["expected"]["params"]["מחיר"] == {"max": 70000}
    assert list(out["spans"]) == ["יצרנים", "מחיר", "צבע"] and out["spans"]["מחיר"][0]["op"] == "max"
    assert out["_normalization"]["renamed"] == [["טווח_מחיר", "מחיר"]]
    assert out["_normalization"]["schema"] == normalize.SCHEMA and not out["_normalization"]["warnings"]


def test_input_not_mutated_and_idempotent():
    row = car_row({"טווח_ק״מ": {"min": 10, "max": 20}})
    snapshot = copy.deepcopy(row)
    once = normalize_row(row)
    assert row == snapshot
    assert normalize_row(once) == once


def test_year_fold_and_ev_range_kept():
    out = normalize_row(car_row({"שנות_ייצור": {"min": 2016}, "טווח_חשמלי_ק״מ": {"min": 400}}))
    assert set(out["expected"]["params"]) == {"שנה", "טווח_חשמלי_ק״מ"}
    assert "רכב/טווח_חשמלי_ק״מ" in normalize.NOT_RENAMED


def test_collision_raises():
    with pytest.raises(NormalizationError):
        normalize_row(car_row({"מחיר": 50000, "טווח_מחיר": {"max": 70000}}))


def test_other_domains_and_negatives_untouched():
    neg = {"id": "n", "query": "שלום", "domain": None, "expected": {"category": None, "params": {}}, "spans": {}}
    assert normalize_row(neg)["expected"] == neg["expected"]
    yad2 = {"id": "y", "query": "q", "domain": "יד_שנייה", "expected": {"params": {"מחיר": {"max": 300}}},
            "spans": {"מחיר": [{"op": "max", "text": "עד 300"}]}}
    assert normalize_row(yad2)["expected"]["params"] == {"מחיר": {"max": 300}}


def test_field_name_mapping():
    assert normalize_field("נדל״ן/טווח_חדרים") == "נדל״ן/מס׳_חדרים"
    assert normalize_field("רכב/טווח_חשמלי_ק״מ") == "רכב/טווח_חשמלי_ק״מ"
    assert normalize_fields(["רכב/שנה", "רכב/שנות_ייצור", "רכב/טווח_שנה"]) == ["רכב/שנה"]


def test_taxonomy_view_drops_exactly_seven_filters():
    from shared.dataset import load_taxonomy

    raw, v2 = load_taxonomy("raw"), load_taxonomy("v2")
    dropped = v2["_normalization"]["dropped_range_filters"]
    assert sum(len(v) for v in dropped.values()) == 7
    assert "פילטרים_נוספים" in raw["קטגוריות"]["רכב"] and "פילטרים_נוספים" not in v2["קטגוריות"]["רכב"]
    assert "טווח_חשמלי_ק״מ" in v2["קטגוריות"]["רכב"]["מאפיינים_כלליים"]


@pytest.mark.skipif(not (DATA / "train.jsonl").exists(), reason="split1 data not present")
def test_real_data_normalizes_cleanly_and_files_are_untouched():
    from shared import dataset

    files = [DATA / f for f in ("train.jsonl", "val.jsonl", "test.jsonl")]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    renamed, warnings = 0, []
    for split in ("train", "val", "test"):
        for r in dataset.load_rows(split, "v2"):
            renamed += len(r["_normalization"]["renamed"])
            warnings += r["_normalization"]["warnings"]
    vocab = dataset.field_vocab("v2")
    assert renamed == 525 and not warnings
    assert len(vocab) == 69 and len(dataset.field_vocab("raw")) == 77
    assert [f for f in vocab if "טווח_" in f] == ["רכב/טווח_חשמלי_ק״מ"]
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
