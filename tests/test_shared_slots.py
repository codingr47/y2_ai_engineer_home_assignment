import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from shared.operators import detect_operator, is_bounded  # noqa: E402
from shared.slots import fields_to_slots, slot_names, slot_vocab, to_slot  # noqa: E402
from shared.taxonomy_rules import Resolver, close_hierarchy, sector_for, storage_field_for, valid_model  # noqa: E402

VOCAB_FILE = ROOT / "training" / "experiment_1" / "artifacts" / "labels" / "field_vocab.json"


def test_twins_map_to_one_slot_and_back():
    s = to_slot("רכב/טווח_מחיר")
    assert s == to_slot("רכב/מחיר") == "רכב/מחיר⇄טווח_מחיר"
    assert slot_names(s) == ("רכב/מחיר", "רכב/טווח_מחיר")
    assert to_slot("רכב/צבע") == "רכב/צבע"


def test_slot_vocab_is_70_when_vocab_available():
    if not VOCAB_FILE.exists():  # labels are regenerable artifacts
        return
    vocab = json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
    assert len(vocab) == 77 and len(slot_vocab(vocab)) == 70  # 7 twin pairs (מחיר in two domains)


def test_point_without_twin_in_domain_stays_a_field():
    vocab = ["יד_שנייה/מחיר", "רכב/מחיר", "רכב/טווח_מחיר"]
    assert fields_to_slots(["יד_שנייה/מחיר"], vocab) == ["יד_שנייה/מחיר"]
    assert fields_to_slots(["רכב/מחיר", "רכב/טווח_מחיר"], vocab) == ["רכב/מחיר⇄טווח_מחיר"]


def test_operator_detector_on_local_spans():
    assert detect_operator("עד 2 מיליון") == "max"
    assert detect_operator("בין 2011 ל-2013") == "range"
    assert detect_operator("4,100,000 ש\"ח ומעלה") == "min"
    assert detect_operator("4 חדרים") == "exact"
    assert detect_operator("לא יותר משתי חניות") == "max"
    assert detect_operator("3 חדרים, בדיוק, לא פחות ולא יותר") == "exact"
    assert is_bounded("range") and not is_bounded("exact")


def test_mixed_operators_resolve_per_field():
    # "מחפש דירת 4 חדרים עד 2 מיליון שקל": each field gets its own local span and operator
    r = Resolver({"נדל״ן/מס׳_חדרים⇄טווח_חדרים": "נדל״ן/מס׳_חדרים", "נדל״ן/מחיר⇄טווח_מחיר": "נדל״ן/מחיר"})
    rooms = r.resolve("נדל״ן/מס׳_חדרים⇄טווח_חדרים", "4 חדרים", detect_operator("4 חדרים"))
    price = r.resolve("נדל״ן/מחיר⇄טווח_מחיר", "עד 2 מיליון שקל", detect_operator("עד 2 מיליון שקל"))
    assert rooms == "נדל״ן/מס׳_חדרים" and price == "נדל״ן/טווח_מחיר"


def test_resolver_default_without_operator():
    r = Resolver.from_train([["רכב/מחיר"], ["רכב/מחיר"], ["רכב/טווח_מחיר"]], ["רכב/מחיר⇄טווח_מחיר", "רכב/צבע"])
    assert r.resolve("רכב/מחיר⇄טווח_מחיר") == "רכב/מחיר"
    assert r.resolve("רכב/צבע") == "רכב/צבע"


def test_hierarchy_closure_and_value_rules():
    assert set(close_hierarchy(["רכב/תתי_דגמים"])) == {"רכב/תתי_דגמים", "רכב/דגמים", "רכב/יצרנים"}
    assert close_hierarchy(["יד_שנייה/תתי_קטגוריות"])[-1] == "יד_שנייה/סקטורים"
    assert storage_field_for("מחשבים_ניידים") == "יד_שנייה/אחסון_GB"
    assert storage_field_for("טלפונים_סלולריים") == "יד_שנייה/נפח_אחסון"
    assert sector_for("טלפונים_סלולריים") == "אלקטרוניקה"
    assert valid_model("טויוטה", "קורולה") is True and valid_model("טויוטה", "סיל") is False
