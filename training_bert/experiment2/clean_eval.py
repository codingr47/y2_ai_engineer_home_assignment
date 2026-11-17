"""2B: the final tagger (experiment-1 ensemble + fallback) on raw vs runtime-cleaned queries.

    python -m training_bert.experiment2.clean_eval [--with-test]

Only rows the cleaner changes are re-predicted (the other rows' inputs are identical, so their predictions are
the saved experiment-1 E12 ones). The three E6 checkpoints run locally (CPU is fine for a few hundred rows).
Writes results/predictions_clean_<split>.jsonl (same format as experiment 1, for the gate) and
results/clean_report.json: JSON exact raw vs cleaned, and on changed rows: fixed / broken / both right / both wrong.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..common import data as D
from ..common.clean import QueryCleaner
from ..common.evaluate import predict, score
from ..common.normalize import SpanNormalizer, dump
from ..final_test import load_run

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "experiment1" / "runs"
MEMBERS = [EXP1 / f"e6_lr5e-5_s{s}" for s in (42, 13, 1337)]


def main(with_test: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cleaner, norm = QueryCleaner.load(), SpanNormalizer.load()
    models = [load_run(p, device)[0] for p in MEMBERS]
    tok = AutoTokenizer.from_pretrained("dicta-il/dictabert")
    report = {}
    for split in ("val_eval", "val_calib") + (("test",) if with_test else ()):
        rows = D.load(split)
        saved = {json.loads(l)["id"]: json.loads(l) for l in
                 (EXP1 / "e12_gate" / f"predictions_gate_{split}.jsonl").read_text(encoding="utf-8").splitlines()}
        cleaned, kinds = [], Counter()
        for r in rows:
            q = cleaner.clean(r["query"])
            kinds.update(k for k, _, _ in cleaner.log)
            cleaned.append({**r, "query": q, "query_raw": r["query"]})
        changed = [r for r in cleaned if r["query"] != r["query_raw"]]
        new = {p["id"]: p for p in predict(models, tok, changed, device, amp=device.type == "cuda")} if changed else {}
        preds = [new.get(r["id"]) or saved[r["id"]]["raw"] for r in cleaned]
        s_raw = score(rows, [saved[r["id"]]["raw"] for r in rows], norm, "spans", fallback="presence")
        s_new = score(cleaned, preds, norm, "spans", fallback="presence", records=True)
        recs = s_new.pop("records")
        with open(HERE / "results" / f"predictions_clean_{split}.jsonl", "w", encoding="utf-8") as f:
            for p, rec, r in zip(preds, recs, cleaned):
                f.write(json.dumps({**rec, "query_raw": r["query_raw"], "raw": p}, ensure_ascii=False) + "\n")
        raw_ok = {r["id"]: bool(saved[r["id"]]["exact"]) for r in rows}
        new_ok = {rec["id"]: bool(rec["exact"]) for rec in recs}
        ch = Counter()
        for r in changed:
            a, b = raw_ok[r["id"]], new_ok[r["id"]]
            ch["fixed" if b and not a else "broken" if a and not b else "both_right" if a else "both_wrong"] += 1
        examples = [{"raw": r["query_raw"], "clean": r["query"], "raw_ok": raw_ok[r["id"]], "clean_ok": new_ok[r["id"]]}
                    for r in changed if raw_ok[r["id"]] != new_ok[r["id"]]]
        report[split] = {"rows": len(rows), "changed_rows": len(changed), "rewrites": dict(kinds),
                         "json_exact_raw": s_raw["json_exact"], "json_exact_clean": s_new["json_exact"],
                         "field_set_raw": s_raw["field_set_exact"], "field_set_clean": s_new["field_set_exact"],
                         "changed_rows_outcome": dict(ch), "flips": examples,
                         "slices_clean": s_new["slices"]}
        print(f"{split}: changed {len(changed)}/{len(rows)} | JSON raw {s_raw['json_exact']:.4f} -> clean "
              f"{s_new['json_exact']:.4f} | fields {s_raw['field_set_exact']:.4f} -> {s_new['field_set_exact']:.4f} "
              f"| changed rows {dict(ch)}", flush=True)
        for e in examples[:8]:
            print("   ", "FIXED " if e["clean_ok"] else "BROKEN", e["raw"][:70], "=>", e["clean"][:70])
    dump(report, HERE / "results" / ("clean_report_with_test.json" if with_test else "clean_report.json"))


if __name__ == "__main__":
    main("--with-test" in sys.argv[1:])
