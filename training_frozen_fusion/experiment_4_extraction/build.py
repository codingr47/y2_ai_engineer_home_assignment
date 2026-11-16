"""Build the extractor's artifacts from TRAIN only (label schema v2, data files untouched).

    python training_frozen_fusion/experiment_4_extraction/build.py [--no-tagger]

Writes src/extraction/artifacts/lexicon_v2.json, taxonomy_v2.json and tagger.joblib (~9 MB, compressed), all
committed as runtime artifacts. The tagger's acceptance threshold is tuned on val_calib by
evaluate_extraction.py --tune-tagger and stored in tagger.threshold.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import paths  # noqa: E402,F401
from shared.dataset import load_rows, load_taxonomy  # noqa: E402
from shared.normalize import SCHEMA  # noqa: E402

from src.extraction.extractor import ARTIFACTS  # noqa: E402
from src.extraction.model import ExtractorModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tagger", action="store_true")
    a = ap.parse_args()
    rows = load_rows("train", "v2")
    tax = load_taxonomy("v2")
    model = ExtractorModel.build(rows, tax, SCHEMA)
    model.save(ARTIFACTS / "lexicon_v2.json")
    (ARTIFACTS / "taxonomy_v2.json").write_text(json.dumps(tax, ensure_ascii=False), encoding="utf-8")
    types = {}
    for fi in model.fields.values():
        types[fi.type] = types.get(fi.type, 0) + 1
    print(f"fields={len(model.fields)} types={types} surfaces={sum(len(f.surfaces) for f in model.fields.values())}")
    if not a.no_tagger:
        from src.extraction.tagger import SpanTagger
        tagger = SpanTagger.train(rows, model)
        tagger.save(ARTIFACTS / "tagger.joblib")
        print(f"tagger: {len(tagger.models)} field models")


if __name__ == "__main__":
    main()
