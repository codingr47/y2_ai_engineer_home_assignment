"""Step 3 inputs: the tagger (cleaned queries, 3-seed ensemble + fallback) and Frozen Fusion on the routing batch.

    python -m training.training_bert.experiment3.routing_predict

The routing batch is calibration-only (its README). It is read as is, labels in schema v2 via normalize_row, and
its partition is NOT redefined here: the fit / selection / audit ids are the ones the existing calibration recorded in
docs/confidence_audit.json (by unit, seed 42). Writes results/routing_tagger.jsonl and results/routing_ff.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.local_parser import public_payload
from training.training_frozen_fusion.shared.normalize import normalize_row

from ..common.clean import QueryCleaner
from ..common.evaluate import predict, score
from ..common.normalize import SpanNormalizer
from ..final_test import load_run
from ..frozen_fusion_baseline import _load_parser

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROUTING = REPO / "training/training_frozen_fusion" / "data" / "fresh_routing_calibration" / "routing.jsonl"
AUDIT_REPORT = REPO / "docs" / "confidence_audit.json"
MEMBERS = [HERE.parent / "experiment1" / "runs" / f"e6_lr5e-5_s{s}" for s in (42, 13, 1337)]
OUT = HERE / "results"


def routing_rows() -> list[dict]:
    rows = [normalize_row(json.loads(l)) for l in ROUTING.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def partition() -> dict[str, set[str]]:
    """The recorded fit / selection / audit ids of the routing batch (not recomputed)."""
    ids = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))["partition_ids"]
    return {k: set(v) for k, v in ids.items()}


def main():
    rows = routing_rows()
    part = partition()
    side = {i: k for k, v in part.items() for i in v}
    assert len(side) == len(rows) and all(r["id"] in side for r in rows), "partition ids must cover the batch"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cleaner, norm = QueryCleaner.load(), SpanNormalizer.load()
    cleaned = [{**r, "query": cleaner.clean(r["query"]), "query_raw": r["query"]} for r in rows]
    models = [load_run(p, device)[0] for p in MEMBERS]
    tok = AutoTokenizer.from_pretrained("dicta-il/dictabert")
    preds = predict(models, tok, cleaned, device, amp=device.type == "cuda")
    s = score(cleaned, preds, norm, "spans", fallback="presence", records=True)
    with open(OUT / "routing_tagger.jsonl", "w", encoding="utf-8") as f:
        for p, rec, r in zip(preds, s.pop("records"), cleaned):
            f.write(json.dumps({**rec, "part": side[r["id"]], "unit": r["unit"], "quality": r.get("quality"),
                                "query_raw": r["query_raw"], "raw": p}, ensure_ascii=False) + "\n")
    print(f"tagger on routing: JSON {s['json_exact']:.4f} fields {s['field_set_exact']:.4f}", flush=True)
    parser = _load_parser()
    with open(OUT / "routing_ff.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            cand = parser.candidate(r["query"])
            f.write(json.dumps({"id": r["id"], "pred": public_payload(cand.payload) if cand.payload else None,
                                "reason": cand.reason, "score": float(parser.gate.score(cand.vector))},
                               ensure_ascii=False) + "\n")
    print("frozen fusion on routing: written", flush=True)


if __name__ == "__main__":
    main()
