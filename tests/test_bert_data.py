"""training_bert data layer: val split, BIO alignment round-trip, span normalizer."""
import pytest

from training_bert.common import data as D
from training_bert.common.normalize import SpanNormalizer, json_equal, reshape


def test_val_split_is_family_disjoint_and_complete():
    calib, ev = D.load("val_calib"), D.load("val_eval")
    assert not ({r["family"] for r in calib} & {r["family"] for r in ev})
    assert len(calib) + len(ev) == len(D.load("val"))


def test_val_split_keeps_frozen_fusion_assignment():
    import json
    ff = json.loads(D.FF_VAL_SPLIT.read_text(encoding="utf-8"))
    ev = {r["id"] for r in D.load("val_eval")}
    present = {r["id"] for r in D.load("val")}
    assert all(i in ev for i in ff["val_eval"] if i in present)
    assert not any(i in ev for i in ff["val_calib"])


def test_subset_is_nested_and_family_grouped():
    train = D.load("train")
    small, big = D.subset(train, 1000, seed=3), D.subset(train, 3000, seed=3)
    assert [r["id"] for r in small] == [r["id"] for r in big[:len(small)]]
    fam = {r["family"] for r in small}
    assert sum(r["family"] in fam for r in train) == len(small)


def test_decode_tags_lenient():
    v = D.Vocab(("נדל״ן/עיר", "נדל״ן/מחיר"), {})
    t = {name: i for i, name in enumerate(v.tags)}
    offs = [(0, 0), (0, 3), (4, 8), (9, 11), (12, 15), (0, 0)]
    tags = [-100, t["B-נדל״ן/עיר"], t["I-נדל״ן/עיר"], t["O"], t["I-נדל״ן/מחיר"], -100]
    assert D.decode_tags(tags, offs, v) == [(0, 8, "נדל״ן/עיר", 1, 3), (12, 15, "נדל״ן/מחיר", 4, 5)]


@pytest.fixture(scope="module")
def tokenizer():
    tr = pytest.importorskip("transformers")
    try:
        return tr.AutoTokenizer.from_pretrained("dicta-il/dictabert", local_files_only=True)
    except OSError:
        pytest.skip("dicta-il/dictabert tokenizer not cached")


def test_bio_roundtrip(tokenizer):
    """Gold char spans -> subword tags -> char spans. Misses are spans that start after a one-letter prefix glued to
    the token (ו/ש/ל/ב), which subwords cannot split."""
    rows = D.load("val_eval")
    vocab = D.Vocab.build(D.load("train"))
    tot = ok = prefix = 0
    for r in rows:
        e = D.encode(r, tokenizer, vocab)
        dec = {(s, t, f) for s, t, f, _, _ in D.decode_tags(e["tags"], e["offsets"], vocab)}
        for s, t, f, _ in D.explicit_spans(r):
            tot += 1
            ok += (s, t, f) in dec
            prefix += any(ff == f and tt == t and s - ss == 1 for ss, tt, ff in dec)
    assert ok / tot > 0.97 and (ok + prefix) / tot > 0.99


@pytest.fixture(scope="module")
def norm():
    return SpanNormalizer.load()


def test_reshape():
    assert reshape(2021, "min") == {"min": 2021}
    assert reshape({"min": 2, "max": 4}, "range") == {"min": 2, "max": 4}
    assert reshape(5, "range") == 5


@pytest.mark.parametrize("field,text,op,value", [
    ("נדל״ן/מחיר", "עד 2 מיליון", "max", {"max": 2000000}),
    ("נדל״ן/מס׳_חדרים", "4 חדרים", "exact", 4),
    ("נדל״ן/חניה", "עם חניה", "min", {"min": 1}),
    ("נדל״ן/ממ״ד", "בלי ממ״ד", "false", False),
    ("נדל״ן/עיר", "בכרמיאל", "value", "כרמיאל"),
])
def test_span_values(norm, field, text, op, value):
    from src.extraction.types import values_equal
    assert values_equal(norm.value(field, text, op), value)


def test_bounds_of_one_field_merge_into_range(norm):
    out = norm.assemble("נדל״ן", [("נדל״ן/מ״ר_בנוי", "מ-75 מ״ר", "min"), ("נדל״ן/מ״ר_בנוי", "עד 100 מ״ר", "max")])
    assert out["params"]["מ״ר_בנוי"] == {"min": 75, "max": 100}


def test_oracle_on_val_eval(norm):
    """Gold spans + gold span-less values through the normalizer: the JSON ceiling for a perfect tagger."""
    rows = D.load("val_eval")
    ok = 0
    for r in rows:
        d, q = r["domain"], r["query"]
        spans = [(f, q[s:e], op) for s, e, f, op in D.explicit_spans(r)]
        implied = {f"{d}/{k}": v for k, v in D.params(r).items() if not (r.get("spans") or {}).get(k)} if d else {}
        ok += json_equal(norm.assemble(d, spans, implied), r["expected"] if d else None)
    assert ok / len(rows) > 0.93


def test_hybrid_vehicle_type_rule():
    from training_bert.common.normalize import HYBRID_VEHICLE_TYPE, hybrid_vehicle_type
    assert hybrid_vehicle_type("רכב", {"סוג_דלק": "היברידי"})["סוגי_רכב"] == HYBRID_VEHICLE_TYPE
    assert hybrid_vehicle_type("רכב", {"סוג_דלק": ["דיזל", "היברידי נטען"]})["סוגי_רכב"] == HYBRID_VEHICLE_TYPE
    assert hybrid_vehicle_type("רכב", {"סוג_דלק": "היברידי", "סוגי_רכב": "רכב שטח"})["סוגי_רכב"] == "רכב שטח"
    assert "סוגי_רכב" not in hybrid_vehicle_type("רכב", {"סוג_דלק": "בנזין"})
    assert hybrid_vehicle_type("יד_שנייה", {"סוג_דלק": "היברידי"}) == {"סוג_דלק": "היברידי"}


def test_hybrid_rule_is_off_by_default(norm):
    out = norm.assemble("רכב", [("רכב/סוג_דלק", "היברידי", "value")])
    assert "סוגי_רכב" not in out["params"]
