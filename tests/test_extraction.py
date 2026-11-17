from pathlib import Path

import pytest

from src.extraction.numbers import parse_numeric
from src.extraction.operators import detect_operator
from src.extraction.text import fold, negated_before, tokens
from src.extraction.types import Extraction, values_equal

ARTIFACTS = Path(__file__).resolve().parents[1] / "src" / "extraction" / "artifacts" / "lexicon_v2.json"
needs_lexicon = pytest.mark.skipif(not ARTIFACTS.exists(), reason="run training/training_frozen_fusion/experiment_4_extraction/build.py")


@pytest.mark.parametrize("text,field,expected", [
    ("שלוש מאות וחמישה אלף", "מחיר", ("exact", 305000)),
    ("1.5 מיליון", "מחיר", ("exact", 1500000)),
    ("עד 220K ק״מ", "ק״מ", ("max", {"max": 220000})),
    ("ארבע וחצי חדרים ומעלה", "מס׳_חדרים", ("min", {"min": 4.5})),
    ("בין 3 ל-4 אלף", "מחיר", ("range", {"min": 3000, "max": 4000})),
    ("מליון וחצי לשניים וחצי מליון", "מחיר", ("range", {"min": 1500000, "max": 2500000})),
    ("2001 סליחה 2010", "שנת_ייצור", ("exact", 2010)),
    ("דונם וחצי", "מ״ר_מגרש", ("exact", 1500)),
    ("1 טרה", "נפח_אחסון", ("exact", 1024)),
    ("יד שלוש עד יד ארבע", "יד", ("range", {"min": 3, "max": 4})),
])
def test_parse_numeric(text, field, expected):
    assert parse_numeric(text, field) == expected


def test_operator_is_span_local():
    assert detect_operator("מנוע עד 3000 סמ״ק") == "max"   # «מנוע» starts with מ but is not «מ<number>»
    assert detect_operator("מנוע מ1400 סמ״ק") == "min"
    assert detect_operator("ועד 195 מ״ר") == "max"
    assert detect_operator("שלא יותר מחמישה חדרים") == "max"


def test_text_helpers():
    assert fold("גיר ידניייי") == "גיר ידני"
    assert fold("נאות-כרמים") == fold("נאות כרמים")
    toks = tokens("לא, רגע ממ״ד")
    assert negated_before(toks, 2) is False           # the correction cancels the negation
    assert negated_before(tokens("בלי ממ״ד"), 1) is True


def test_values_equal():
    assert values_equal({"min": 3, "max": 4.0}, {"max": 4, "min": 3})
    assert values_equal(["כסוף", "כחול"], ["כחול", "כסוף"])
    assert values_equal('ק"מ', "ק״מ")
    assert not values_equal(True, 1)
    assert not values_equal({"min": 3}, {"max": 3})


@pytest.fixture(scope="module")
def ex():
    from src.extraction.extractor import FieldExtractor

    return FieldExtractor.load()


@needs_lexicon
def test_mixed_operators_are_field_local(ex):
    q = "מחפש דירת 4 חדרים עד 2 מיליון שקל"
    out = ex.extract_all(q, "נדל״ן", ["נדל״ן/מס׳_חדרים", "נדל״ן/מחיר"])
    assert out["נדל״ן/מס׳_חדרים"].value == 4 and out["נדל״ן/מס׳_חדרים"].operator == "exact"
    assert out["נדל״ן/מחיר"].value == {"max": 2000000} and out["נדל״ן/מחיר"].operator == "max"


@needs_lexicon
def test_city_with_prefix_and_boolean_negation(ex):
    out = ex.extract_all("מה אפשר לשכור ברחובות בלי ממ״ד", "נדל״ן", ["נדל״ן/עיר", "נדל״ן/ממ״ד"])
    assert out["נדל״ן/עיר"].value == "רחובות"
    assert out["נדל״ן/ממ״ד"].value is False


@needs_lexicon
def test_list_date_and_taxonomy_derivation(ex):
    assert ex.extract("רכב היברידי או חשמלי", "רכב/סוג_דלק").value in (["היברידי", "חשמלי"], ["חשמלי", "היברידי"])
    assert ex.extract("כניסה בתאריך 22 במאי 2027", "נדל״ן/תאריך_כניסה").value == "22-05-2027"
    out = ex.extract_all("קורולה לבנה", "רכב", ["רכב/דגמים", "רכב/יצרנים"])
    assert out["רכב/דגמים"].value == "קורולה" and out["רכב/יצרנים"].value == "טויוטה"


@needs_lexicon
def test_numbers_are_not_claimed_twice_and_json_assembly(ex):
    q = "3 חדרים קומה 2"
    out = ex.extract_all(q, "נדל״ן", ["נדל״ן/מס׳_חדרים", "נדל״ן/קומה"])
    spans = [(e.start, e.end) for e in out.values() if e.found]
    assert len(spans) == len(set(spans))
    assert out["נדל״ן/מס׳_חדרים"].value == 3 and out["נדל״ן/קומה"].value == 2
    js = ex.to_json("נדל״ן", {**out, "נדל״ן/מעלית": Extraction("נדל״ן/מעלית", reason="no_span")})
    assert js == {"category": "נדל״ן", "params": {"מס׳_חדרים": 3, "קומה": 2}}
    assert ex.to_json("none", out) is None


@needs_lexicon
@pytest.mark.parametrize("query,field,expected", [
    ("אפשר לשכור בלי בעלי חיים?", "נדל״ן/חיות_מחמד", False),       # negation inside the learned surface
    ("משהו קטן בלימיזוג", "נדל״ן/מיזוג", False),                    # glued negation
    ("עד 2 מרפסות. בליי מרפסת שמש", "נדל״ן/מרפסת_שמש", False),     # typo'd negation
    ("דירה עם מעלית", "נדל״ן/מעלית", True),
])
def test_boolean_negation_forms(ex, query, field, expected):
    assert ex.extract_all(query, field.split("/")[0], [field])[field].value is expected


@needs_lexicon
@pytest.mark.parametrize("query,field,expected", [
    ("דירה בבניין לשימור בין 145 ל-160 מ״ר בנוי", "נדל״ן/מ״ר_בנוי", {"min": 145, "max": 160}),
    ("בקומה ארבע עשרה עד עשרים שמור", "נדל״ן/קומה", {"min": 14, "max": 20}),
    ("כאילו משהו בין קומה 0 לקומה 5", "נדל״ן/קומה", {"min": 0, "max": 5}),
    ("ארנונה בין אלף חמישים לאלף ארבע מאות חמישים", "נדל״ן/ארנונה_חודשית", {"min": 1050, "max": 1450}),
    ("לפטופ עם Ryzen 5 ולפחות 32 גיגה ראם", "יד_שנייה/זיכרון_RAM", {"min": 32}),
    ("חצי מרוהט ועם 1+ חניות", "נדל״ן/חניה", {"min": 1}),
    ("מנוע מאלף סנטימטר מעוקב ומעלה", "רכב/נפח_מנוע_סמ״ק", {"min": 1000}),
    ("טלוויזיה בבאר שבע עד 65 אינץ'", "יד_שנייה/גודל_אינצ׳", {"max": 65}),   # «שבע» is the city, not 7
    ("דירת 4 חדרים עד 2 מיליון", "נדל״ן/מחיר", {"max": 2000000}),            # 4 .. 2M is not one range
])
def test_numeric_ranges_and_bounds(ex, query, field, expected):
    assert ex.extract_all(query, field.split("/")[0], [field])[field].value == expected


@needs_lexicon
@pytest.mark.parametrize("query,field,expected", [
    ("מכשיר סלולרי במצב כום חדש", "יד_שנייה/מצב", "כמו חדש"),     # typo'd longer value beats the exact shorter one
    ("קורולה SE היברידי נטעם", "רכב/סוג_דלק", "היברידי נטען"),
    ("טלפון במצב חדש", "יד_שנייה/מצב", "חדש"),
    ("טויוטה קורולה בגרסת LE", "רכב/תתי_דגמים", "LE"),             # LE never becomes XLE
])
def test_longer_typo_values(ex, query, field, expected):
    assert ex.extract_all(query, field.split("/")[0], [field])[field].value == expected
