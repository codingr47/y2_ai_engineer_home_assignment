"""Error analysis of a scored run: why rows miss JSON exact, per field and per cause.

    python -m training.training_bert.analyze_errors training/training_bert/experiment1/runs/e6_lr5e-5_s42 [--tag=norm2] [--split=val_eval]

Reads <run>/predictions<_tag>_<split>.jsonl (written by final_test) and prints / writes errors<_tag>_<split>.json:
- stage of the first failure (domain / field_set / values)
- field-set errors split into missed and spurious fields, per field, and whether the presence head knew
  (a missed field the presence head scored >= 0.5 had no span; a spurious field it scored < 0.5 came from a span)
- value errors per field type
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .common import data as D
from .common.normalize import SpanNormalizer, dump


def main(run: Path, tag: str = "", split: str = "val_eval"):
    sfx = f"_{tag}" if tag else ""
    recs = [json.loads(l) for l in (run / f"predictions{sfx}_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    norm = SpanNormalizer.load()
    stage = Counter(r["stage"] for r in recs)
    missed, spurious, wrong = Counter(), Counter(), Counter()
    missed_presence, spurious_presence = Counter(), Counter()
    examples = defaultdict(list)
    for r in recs:
        if r["stage"] in ("ok", "domain"):
            continue
        d = r["domain"]
        gold, pred = r["gold"] or {}, r["pred"] or {}
        pres = (r.get("raw") or {}).get("present", {})
        for k in set(gold) - set(pred):
            f = f"{d}/{k}"
            missed[f] += 1
            missed_presence["presence>=0.5 (no span found)" if pres.get(f, 0) >= .5 else "presence<0.5 too"] += 1
            if len(examples["missed " + f]) < 4:
                examples["missed " + f].append(r["query"])
        for k in set(pred) - set(gold):
            f = f"{d}/{k}"
            spurious[f] += 1
            spurious_presence["presence<0.5 (span alone)" if pres.get(f, 0) < .5 else "presence>=0.5 too"] += 1
            if len(examples["spurious " + f]) < 4:
                examples["spurious " + f].append([r["query"], pred[k]])
        if r["stage"] == "values":
            from src.extraction.types import values_equal
            for k in set(gold) & set(pred):
                if not values_equal(gold[k], pred[k]):
                    wrong[(norm.ftype(f"{d}/{k}"), k)] += 1
                    if len(examples[f"wrong {d}/{k}"]) < 4:
                        examples[f"wrong {d}/{k}"].append([r["query"], gold[k], pred[k]])
    n = len(recs)
    out = {"run": run.name, "split": split, "n": n, "stage": {k: v / n for k, v in stage.items()},
           "missed_fields": missed.most_common(25), "missed_by_presence": dict(missed_presence),
           "spurious_fields": spurious.most_common(25), "spurious_by_presence": dict(spurious_presence),
           "wrong_values": [[list(k), v] for k, v in wrong.most_common(25)], "examples": examples}
    dump(out, run / f"errors{sfx}_{split}.json")
    print(json.dumps({k: out[k] for k in ("stage", "missed_by_presence", "spurious_by_presence")}, ensure_ascii=False))
    print("missed  ", missed.most_common(12))
    print("spurious", spurious.most_common(12))
    print("wrong   ", wrong.most_common(10))
    for k in [f"missed {f}" for f, _ in missed.most_common(5)] + [f"spurious {f}" for f, _ in spurious.most_common(5)]:
        print(k, examples[k][:3])


if __name__ == "__main__":
    a = sys.argv[1:]
    tag = next((x.split("=", 1)[1] for x in a if x.startswith("--tag=")), "")
    split = next((x.split("=", 1)[1] for x in a if x.startswith("--split=")), "val_eval")
    main(Path(next(x for x in a if not x.startswith("--"))), tag, split)
