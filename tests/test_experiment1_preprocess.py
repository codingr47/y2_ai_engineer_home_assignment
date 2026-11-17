import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/training_frozen_fusion" / "experiment_1"))

from preprocess import split_val, to_label  # noqa: E402


def test_label_shape_matches_spec():
    row = {"id": "syn_0023", "query": "מחפש קלדים שעולה 4200 שקל", "domain": "יד_שנייה",
           "expected": {"category": "יד_שנייה",
                        "params": {"סקטורים": "מוסיקה_וכלים", "תתי_קטגוריות": "קלידים", "מחיר": 4200}}}
    assert to_label(row) == {"id": "syn_0023", "domain_label": "יד_שנייה",
                             "field_labels": ["יד_שנייה/סקטורים", "יד_שנייה/תתי_קטגוריות", "יד_שנייה/מחיר"]}


def test_null_domain_is_none_with_no_fields():
    row = {"id": "syn_9", "query": "איך מפרסמים מודעה", "domain": None, "expected": {"category": None, "params": {}}}
    assert to_label(row) == {"id": "syn_9", "domain_label": "none", "field_labels": []}


def test_gershayim_domain_kept_verbatim():
    row = {"id": "x", "query": "q", "domain": "נדל״ן", "expected": {"params": {"חדרים": 3}}}
    assert to_label(row)["field_labels"] == ["נדל״ן/חדרים"]


def test_val_split_is_family_grouped_and_halved():
    rows = [{"id": f"r{i}", "query": "q", "family": f"f{i // 2}", "domain": ["רכב", "נדל״ן", None][i % 3],
             "expected": {"params": {}}} for i in range(120)]
    a, b = split_val(rows)
    assert len(a) + len(b) == 120 and abs(len(a) - len(b)) <= 12
    assert not {r["family"] for r in a} & {r["family"] for r in b}
