"""Export the frozen training bundle without training-owned pickle classes.

Run: python -m scripts.export_classifier
"""
import argparse
from pathlib import Path

import joblib
import numpy as np


def export(bundle, destination: Path):
    if bundle.label_space != "v2" or bundle.kind != "e5+tfidf":
        raise ValueError("expected frozen Exp3 v2 fusion bundle")
    h, f = bundle.heads, bundle.featurizer
    artifact = {
        "schema": "exp3-v2-runtime-1", "word": f.tfidf.word, "char": f.tfidf.char,
        "weight": f.w, "vocab": h.vocab, "classes": h.domain_model.classes_,
        "domain_model": h.domain_model, "domain_method": h.domain_cal.method,
        "field_model": h.field_model, "platt_a": h.field_cal.a, "platt_b": h.field_cal.b,
        "thresholds": h.field_cal.thr, "encoder": "intfloat/multilingual-e5-small",
        "embedding_dimension": 384, "query_prefix": "query: ",
    }
    if h.domain_cal.method == "temperature":
        artifact["temperature"] = h.domain_cal.cal.T
    else:
        artifact["domain_calibrator"] = h.domain_cal.cal
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination, compress=3)
    return artifact


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=Path("training_frozen_fusion/experiment_3/artifacts/models_v2/bundle.joblib"))
    ap.add_argument("--output", type=Path, default=Path("src/artifacts/classifier.joblib"))
    args = ap.parse_args()
    from training_frozen_fusion.shared import paths  # registers legacy training pickle modules
    from src.classifier import FusionClassifier
    from training_frozen_fusion.shared.runner import base_split
    bundle = joblib.load(args.bundle)
    artifact = export(bundle, args.output)
    X = bundle.featurizer.transform(base_split("val_eval"))
    original = bundle.predict_X(X)
    exported = FusionClassifier(artifact, None).predict_X(X)
    for old, new in zip(original, exported):
        np.testing.assert_allclose(old, new, atol=1e-12, rtol=1e-12)
    print(f"Exported {args.output}; full val_eval probabilities and selected fields match")


if __name__ == "__main__":
    main()
