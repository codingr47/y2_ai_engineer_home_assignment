"""Fit, select, and independently audit whole-parse confidence on a new labeled handoff.

python -m scripts.calibrate_confidence --data routing.jsonl --output src/artifacts/confidence.joblib
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import unicodedata

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.confidence import FEATURE_NAMES, ConfidenceGate, bindings
from src.extraction.extractor import FieldExtractor
from src.extraction.text import clean
from src.extraction.types import values_equal
from src.local_parser import LocalParser, public_payload, validate_output
from src.settings import Settings


class RoutingRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=2000)
    domain: str | None
    expected: dict
    group_id: str = Field(min_length=1)
    language: str = "unknown"


def query_key(query):
    # Preserve numbers and spelling: extractor matching folds would conflate 111 with 1.
    return " ".join(clean(unicodedata.normalize("NFKC", query)).casefold().split())


def partition(rows, existing_queries):
    existing = {query_key(q) for q in existing_queries}
    ids, queries, groups = set(), set(), defaultdict(list)
    for row in rows:
        key = query_key(row.query)
        if not key or row.query != row.query.strip():
            raise ValueError("routing queries must be nonblank and trimmed")
        if key in existing or key in queries or row.id in ids:
            raise ValueError("duplicate routing ID/query or overlap with existing data")
        ids.add(row.id)
        queries.add(key)
        groups[row.group_id].append(row)
    keys = sorted(groups)
    if len(keys) < 5:
        raise ValueError("at least five independent query groups are required")
    random.Random(42).shuffle(keys)
    fit_end, selection_end = int(.6 * len(keys)), int(.8 * len(keys))
    return [[row for key in subset for row in groups[key]] for subset in
            (keys[:fit_end], keys[fit_end:selection_end], keys[selection_end:])]


def wilson_lower(correct, n):
    if n == 0:
        return 0.0
    z = 1.6448536269514722  # one-sided 95% lower confidence bound
    p = correct / n
    return float((p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n))


def operating_point(scores, correct, eligible, threshold):
    accepted = eligible & (scores >= threshold) if threshold is not None else np.zeros(len(scores), dtype=bool)
    n = int(accepted.sum())
    k = int(correct[accepted].sum())
    return {"n": len(scores), "accepted": n, "correct": k,
            "accuracy": k / n if n else None, "coverage": n / len(scores) if len(scores) else 0.0,
            "fallback_rate": 1 - n / len(scores) if len(scores) else 1.0,
            "accuracy_lower_95": wilson_lower(k, n)}


def select_threshold(scores, correct, eligible):
    best = None
    best_n = 0
    for threshold in sorted(set(scores[eligible])):
        result = operating_point(scores, correct, eligible, threshold)
        if result["accuracy_lower_95"] >= .95 and result["accepted"] > best_n:
            best, best_n = float(threshold), result["accepted"]
    return best


def labeled_features(parser, rows):
    X, y, eligible = [], [], []
    for row in rows:
        if set(row.expected) != {"params"} or not isinstance(row.expected["params"], dict):
            raise ValueError("expected must contain only a params object")
        gold = None if row.domain is None else {"category": row.domain, "params": row.expected["params"]}
        if gold:
            validate_output(gold, parser.extractor)
        elif row.expected["params"]:
            raise ValueError("non-search rows must have empty params")
        candidate = parser.candidate(row.query)
        pred = public_payload(candidate.payload) if candidate.payload else None
        expected = public_payload(gold) if gold else None
        correct = pred is not None and expected is not None and pred["category"] == expected["category"]
        if correct:
            correct = set(pred["params"]) == set(expected["params"]) and all(
                values_equal(pred["params"][k], v) for k, v in expected["params"].items())
        X.append(candidate.vector)
        y.append(bool(correct))
        eligible.append(candidate.reason is None)
    return np.array(X), np.array(y), np.array(eligible)


def slices(rows, scores, y, eligible, threshold):
    output = {}
    for attribute in ("domain", "language"):
        output[attribute] = {}
        for value in sorted({str(getattr(r, attribute)) for r in rows}):
            mask = np.array([str(getattr(r, attribute)) == value for r in rows])
            output[attribute][value] = operating_point(scores[mask], y[mask], eligible[mask], threshold)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("src/artifacts/confidence.joblib"))
    ap.add_argument("--report", type=Path, default=Path("docs/confidence_audit.json"))
    args = ap.parse_args()
    settings = Settings.from_env()
    rows = [RoutingRow.model_validate(json.loads(line)) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    from training.shared.dataset import load_rows
    old_queries = [r["query"] for split in ("train", "val", "test") for r in load_rows(split)]
    fit_rows, select_rows, audit_rows = partition(rows, old_queries)
    from src.classifier import FusionClassifier
    classifier = FusionClassifier.load(settings.classifier_artifact_path, settings.e5_model_path)
    extractor = FieldExtractor.load(use_tagger=settings.extractor_use_tagger)
    parser = LocalParser(classifier, extractor, ConfidenceGate())
    X, y, _ = labeled_features(parser, fit_rows)
    if len(set(y)) != 2:
        raise ValueError("confidence fit partition requires both correct and incorrect parses")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=42)).fit(X, y)
    Xs, ys, es = labeled_features(parser, select_rows)
    ps = model.predict_proba(Xs)[:, 1]
    threshold = select_threshold(ps, ys, es)
    Xa, ya, ea = labeled_features(parser, audit_rows)
    pa = model.predict_proba(Xa)[:, 1]
    audit = operating_point(pa, ya, ea, threshold)
    enabled = threshold is not None and audit["accuracy_lower_95"] >= .95
    bound = bindings(settings.classifier_artifact_path, settings.e5_model_path, settings.extractor_use_tagger)
    report = {"schema": "parse-confidence-1", "target_accuracy": .95, "acceptance_enabled": enabled,
              "threshold": threshold, "fit_n": len(fit_rows), "selection": operating_point(ps, ys, es, threshold),
              "audit": audit, "audit_slices": slices(audit_rows, pa, ya, ea, threshold),
              "selection_slices": slices(select_rows, ps, ys, es, threshold),
              "split_seed": 42, "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "partition_ids": {k: [r.id for r in v] for k, v in
                                (("fit", fit_rows), ("selection", select_rows), ("audit", audit_rows))},
              "bindings": bound,
              "limitation": "Wilson bounds assume independent queries; correlated variants must share group_id. "
                            "Reliability applies to the supplied data distribution, not arbitrary production traffic."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**report, "features": FEATURE_NAMES, "model": model}, args.output, compress=3)
    print(f"acceptance_enabled={enabled}; threshold={threshold}; audit={audit}")
    return 0 if enabled else 2


if __name__ == "__main__":
    raise SystemExit(main())
