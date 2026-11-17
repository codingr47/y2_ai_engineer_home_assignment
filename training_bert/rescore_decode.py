"""Re-decode saved raw predictions under different decoding options (no model, CPU only).

    python -m training_bert.rescore_decode training_bert/experiment1/runs/e6_lr5e-5_s42 --tag=norm2 [--split=val_eval]

Reads <run>/predictions<_tag>_<split>.jsonl ("raw" = the model's predictions) and scores every combination of
fallback (none / span / presence) and presence threshold, writing <run>/decode<_tag>_<split>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .common import data as D
from .common.evaluate import score
from .common.normalize import SpanNormalizer, dump

FALLBACKS = ("none", "span", "presence")
THRESHOLDS = (0.3, 0.5, 0.7)


def main(run: Path, tag: str = "", split: str = "val_eval"):
    sfx = f"_{tag}" if tag else ""
    raw = [json.loads(l)["raw"] for l in (run / f"predictions{sfx}_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = D.load(split)
    norm = SpanNormalizer.load()
    out = {}
    for fb in FALLBACKS:
        for th in (THRESHOLDS if fb == "presence" else (0.5,)):
            s = score(rows, raw, norm, "spans", threshold=th, fallback=fb)
            key = f"{fb}@{th}"
            out[key] = {k: s[k] for k in ("json_exact", "field_set_exact", "value_acc_on_shared_fields", "waterfall")}
            print(f"{key:14s} JSON {s['json_exact']:.4f} fields {s['field_set_exact']:.4f} "
                  f"waterfall {json.dumps({k: round(v, 4) for k, v in s['waterfall'].items()})}", flush=True)
    dump(out, run / f"decode{sfx}_{split}.json")


if __name__ == "__main__":
    a = sys.argv[1:]
    tag = next((x.split("=", 1)[1] for x in a if x.startswith("--tag=")), "")
    split = next((x.split("=", 1)[1] for x in a if x.startswith("--split=")), "val_eval")
    main(Path(next(x for x in a if not x.startswith("--"))), tag, split)
