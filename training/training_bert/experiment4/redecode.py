"""Experiment 4: decoding fixes F1-F3 measured on val by re-decoding the saved model outputs (no model run, no test).

    python -m training.training_bert.experiment4.redecode

Input: experiment2/results/predictions_clean_<split>.jsonl. These hold the final ensemble's raw outputs ("raw") on
runtime-cleaned queries for val_eval and val_calib; Frozen Fusion's parses are in experiment2/results/ff_<split>.jsonl.

Fixes (src/tagger/normalize.py):
    F1  storage units glued to their number («1TB», «512GB») are split before reading
    F2  canonical names up to hyphens / spaces / a glued one-letter prefix («נאות-כרמים» = «נאות כרמים»)
    F3  one property type, the most specific («דירת גג» -> גג/טיפוס, not [דירה, גג/טיפוס])
The ablation switches each fix off in turn. With all fixes on, results/predictions_<split>.jsonl is written in the
experiment-2 record format, so the gate can be refitted on the decoding that is actually served
(scripts/build_tagger_gate.py --decoded experiment4).
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from src.tagger import normalize as RT

from ..common import data as D
from ..common.evaluate import score
from ..common.normalize import SpanNormalizer, dump

HERE = Path(__file__).resolve().parent
E2 = HERE.parent / "experiment2" / "results"
OUT = HERE / "results"
SPLITS = ("val_eval", "val_calib")


@contextmanager
def without(fix: str, norm: SpanNormalizer):
    """Temporarily switch one fix off."""
    saved = (RT._GLUED_UNIT, norm.canonical, RT.GENERIC_PROPERTY_TYPES)
    if fix == "F1":
        RT._GLUED_UNIT = __import__("re").compile(r"(?!x)x")          # never matches
    elif fix == "F2":
        norm.canonical = lambda field, value: value
    elif fix == "F3":
        RT.GENERIC_PROPERTY_TYPES = frozenset()
    try:
        yield
    finally:
        RT._GLUED_UNIT, norm.canonical, RT.GENERIC_PROPERTY_TYPES = saved


def load(split):
    recs = [json.loads(l) for l in (E2 / f"predictions_clean_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = {r["id"]: r for r in D.load(split)}
    cleaned = [{**rows[r["id"]], "query": r["query"]} for r in recs]      # the query the tagger read (cleaned)
    return recs, cleaned, [r["raw"] for r in recs]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    norm = SpanNormalizer.load()
    data = {s: load(s) for s in SPLITS}
    table = {}
    for variant in ("all_off", "F1_off", "F2_off", "F3_off", "all_on"):
        offs = {"all_off": ("F1", "F2", "F3"), "all_on": ()}.get(variant, (variant[:2],))
        res = {}
        with without(offs[0], norm) if len(offs) > 0 else contextmanager(lambda: (yield))():
            with without(offs[1], norm) if len(offs) > 1 else contextmanager(lambda: (yield))():
                with without(offs[2], norm) if len(offs) > 2 else contextmanager(lambda: (yield))():
                    for s in SPLITS:
                        recs, cleaned, raw = data[s]
                        sc = score(cleaned, raw, norm, "spans", fallback="presence", records=(variant == "all_on"))
                        res[s] = {"json_exact": sc["json_exact"], "field_set_exact": sc["field_set_exact"]}
                        if variant == "all_on":
                            with open(OUT / f"predictions_{s}.jsonl", "w", encoding="utf-8") as f:
                                for rec, new in zip(recs, sc["records"]):
                                    f.write(json.dumps({**new, "query_raw": rec["query_raw"], "raw": rec["raw"]},
                                                       ensure_ascii=False) + "\n")
        table[variant] = res
        print(f"{variant:8s} " + " | ".join(f"{s} JSON {v['json_exact']:.4f} fields {v['field_set_exact']:.4f}"
                                             for s, v in res.items()), flush=True)
    base = table["all_off"]
    gains = {v: {s: round(table[v][s]["json_exact"] - base[s]["json_exact"], 4) for s in SPLITS} for v in table}
    dump({"variants": table, "json_gain_vs_all_off": gains,
          "note": "val only; experiment-2 raw model outputs re-decoded; test not used"}, OUT / "redecode_report.json")


if __name__ == "__main__":
    main()
