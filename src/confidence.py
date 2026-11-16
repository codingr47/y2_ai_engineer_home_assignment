"""Whole-parse confidence features and version-bound routing artifacts."""
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np

from src.extraction.extractor import ARTIFACTS
from src.extraction.types import SOURCES

FEATURE_NAMES = (
    "domain_confidence", "domain_margin", "min_selected", "max_rejected", "prediction_margin",
    "threshold_distance", "selected_count", "missing_count", "found_count", "score_min", "score_mean",
    *("source_" + s for s in SOURCES),
)


def features(classification, extractions):
    c = classification
    dp = sorted(c.domain_probs, reverse=True)
    selected = c.field_probs[c.selected]
    rejected = c.field_probs[c.in_domain & ~c.selected]
    minimum = float(selected.min()) if len(selected) else 1.0
    maximum = float(rejected.max()) if len(rejected) else 0.0
    distances = np.abs(c.field_probs[c.in_domain] - c.thresholds[c.in_domain])
    found = [e for e in extractions.values() if e.found]
    scores = [e.score for e in found]
    values = [dp[0], dp[0] - dp[1], minimum, maximum, minimum - maximum,
              float(distances.min()) if len(distances) else 0.0, len(c.fields),
              sum(not e.found for e in extractions.values()), len(found),
              min(scores, default=0.0), float(np.mean(scores)) if scores else 0.0,
              *(sum(e.source == s for e in found) for s in SOURCES)]
    vector = np.array(values, dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError("non-finite confidence features")
    return vector


def bindings(classifier_path: Path, encoder_path: Path, use_tagger: bool):
    """Bind decisions to model assets, encoder content, inference code, and feature schema."""
    def digest(paths):
        h = hashlib.sha256()
        for p in sorted(paths, key=lambda p: str(p)):
            h.update(p.name.encode())
            with p.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
        return h.hexdigest()

    root = Path(__file__).resolve().parent
    assets = [ARTIFACTS / "lexicon_v2.json", ARTIFACTS / "taxonomy_v2.json"]
    if use_tagger:
        assets += [ARTIFACTS / "tagger.joblib", ARTIFACTS / "tagger.threshold.json"]
    return {
        "classifier": digest([classifier_path]), "extractor": digest(assets),
        "encoder": digest([p for p in encoder_path.rglob("*") if p.is_file() and ".cache" not in p.parts]),
        "code": digest([root / "classifier.py", root / "local_parser.py", root / "confidence.py", root / "models.py",
                        *sorted((root / "extraction").glob("*.py"))]),
        "features": hashlib.sha256(json.dumps(FEATURE_NAMES).encode()).hexdigest(),
        "use_tagger": use_tagger,
    }


class ConfidenceGate:
    def __init__(self, artifact: dict | None = None, threshold: float | None = None):
        self.artifact = artifact
        self.enabled = bool(artifact and artifact.get("acceptance_enabled"))
        self.threshold = threshold if threshold is not None else (artifact or {}).get("threshold")
        if self.threshold is not None and (not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1):
            raise ValueError("invalid confidence threshold")

    @classmethod
    def load(cls, path: Path, expected_bindings: dict, threshold=None):
        if not path.exists():
            return cls(threshold=threshold)
        artifact = joblib.load(path)
        if artifact.get("schema") != "parse-confidence-1" or artifact.get("bindings") != expected_bindings:
            raise ValueError("confidence artifact does not match the active parser")
        if tuple(artifact.get("features", ())) != FEATURE_NAMES:
            raise ValueError("confidence feature schema mismatch")
        if artifact.get("acceptance_enabled") and artifact.get("threshold") is None:
            raise ValueError("acceptance-enabled confidence artifact requires an audited threshold")
        return cls(artifact, threshold)

    def score(self, vector):
        if self.artifact is None:
            return None
        score = float(self.artifact["model"].predict_proba([vector])[0, 1])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid confidence model output")
        return score

    def accepts(self, score):
        return self.enabled and score is not None and self.threshold is not None and score >= self.threshold
