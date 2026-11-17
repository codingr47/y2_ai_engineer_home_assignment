"""Frozen Fusion's per-row parse on val_eval / val_calib / test, as a second opinion for the tagger's gate (2A).

    python -m training_bert.experiment2.ff_predictions        -> experiment2/results/ff_<split>.jsonl

Uses the shipped parser in src/artifacts (classifier, extractor, its own gate score), loaded with the tolerant
code binding of frozen_fusion_baseline (only docstrings changed on main). Frozen Fusion never trained on these rows.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.local_parser import public_payload

from ..common import data as D
from ..frozen_fusion_baseline import _load_parser

OUT = Path(__file__).resolve().parent / "results"


def main():
    parser = _load_parser()
    for split in ("val_eval", "val_calib", "test"):
        with open(OUT / f"ff_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in D.load(split):
                cand = parser.candidate(r["query"])
                f.write(json.dumps({"id": r["id"], "pred": public_payload(cand.payload) if cand.payload else None,
                                    "reason": cand.reason, "score": float(parser.gate.score(cand.vector)),
                                    "domain": parser.classifier.classify(r["query"]).domain},
                                   ensure_ascii=False) + "\n")
        print("wrote", split, flush=True)


if __name__ == "__main__":
    main()
