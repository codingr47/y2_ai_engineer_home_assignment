"""Fit, select, and independently audit whole-parse confidence on a new labeled handoff.

python -m scripts.calibrate_confidence --data training_frozen_fusion/data/fresh_routing_calibration/routing.jsonl

--data takes either the generator's routing export (rows carry `unit`; labels use the original field names and are
converted to label schema v2 here) or rows already in RoutingRow form. Partitions are by group (the export's
`unit`), and reliability is counted per group: rows in one group (a query and its rephrasing) are not independent.
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
    quality: str = "unknown"
    label_seen_before: str = "unknown"


def from_export(raw):
    """A generator routing-export row -> RoutingRow, with labels in schema v2 (the files are never modified)."""
    from training_frozen_fusion.shared.normalize import normalize_row
    row = normalize_row(raw)
    expected = row["expected"]
    if expected.get("category") != row["domain"]:
        raise ValueError(f"{row['id']}: expected.category disagrees with domain")
    meta = row.get("metadata") or {}
    return RoutingRow(id=row["id"], query=row["query"], domain=row["domain"],
                      expected={"params": expected["params"]}, group_id=row["unit"],
                      language="+".join(meta.get("language") or ["unknown"]), quality=row.get("quality", "unknown"),
                      label_seen_before=str(meta.get("label_seen_before", "unknown")).lower())


def read_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            rows.append(from_export(raw) if "unit" in raw else RoutingRow.model_validate(raw))
    return rows


def query_key(query):
    # Preserve numbers and spelling: extractor matching folds would conflate 111 with 1.
    return " ".join(clean(unicodedata.normalize("NFKC", query)).casefold().split())


def partition(rows, existing_queries):
    existing = {query_key(q) for q in existing_queries}
    ids, queries, groups = set(), set(), defaultdict(list)
    for row in rows:
        key = query_key(row.query)
        if not key:  # surrounding whitespace is kept: it is part of the intentional noise users type
            raise ValueError("routing queries must be nonblank")
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


def operating_point(scores, correct, eligible, groups, threshold):
    """Accuracy of accepted parses. The lower bound counts independent groups, not rows: the effective sample size is
    the number of accepted groups, at the row-level accuracy (rephrasings of one query are correlated)."""
    accepted = eligible & (scores >= threshold) if threshold is not None else np.zeros(len(scores), dtype=bool)
    n = int(accepted.sum())
    k = int(correct[accepted].sum())
    n_groups = len(set(groups[accepted]))
    accuracy = k / n if n else None
    return {"n": len(scores), "accepted": n, "correct": k, "accepted_groups": n_groups,
            "accuracy": accuracy, "coverage": n / len(scores) if len(scores) else 0.0,
            "fallback_rate": 1 - n / len(scores) if len(scores) else 1.0,
            "accuracy_lower_95": wilson_lower(accuracy * n_groups, n_groups) if n else 0.0,
            "accuracy_lower_95_rows": wilson_lower(k, n)}


def select_threshold(scores, correct, eligible, groups, target):
    """Largest coverage whose lower bound on the selection part reaches `target`. The target sits above the audit
    target: picking the threshold that only just passes on selection is optimistic and would usually fail the audit."""
    best = None
    best_n = 0
    for threshold in sorted(set(scores[eligible])):
        result = operating_point(scores, correct, eligible, groups, threshold)
        if result["accuracy_lower_95"] >= target and result["accepted"] > best_n:
            best, best_n = float(threshold), result["accepted"]
    return best


def labeled_features(parser, rows):
    """Features, whole-parse correctness and eligibility. A gold value outside the extraction schema (a city or colour
    the extractor has never seen) is kept: the local parser cannot produce it, so the row counts as incorrect."""
    X, y, eligible, outside = [], [], [], 0
    for row in rows:
        if set(row.expected) != {"params"} or not isinstance(row.expected["params"], dict):
            raise ValueError("expected must contain only a params object")
        gold = None if row.domain is None else {"category": row.domain, "params": row.expected["params"]}
        if gold:
            try:
                validate_output(gold, parser.extractor)
            except ValueError:
                outside += 1
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
    groups = np.array([r.group_id for r in rows])
    return np.array(X), np.array(y), np.array(eligible), groups, outside


def slices(rows, scores, y, eligible, groups, threshold):
    output = {}
    for attribute in ("domain", "language", "quality", "label_seen_before"):
        output[attribute] = {}
        for value in sorted({str(getattr(r, attribute)) for r in rows}):
            mask = np.array([str(getattr(r, attribute)) == value for r in rows])
            output[attribute][value] = operating_point(scores[mask], y[mask], eligible[mask], groups[mask], threshold)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("src/artifacts/confidence.joblib"))
    ap.add_argument("--report", type=Path, default=Path("docs/confidence_audit.json"))
    ap.add_argument("--target", type=float, default=.95,
                    help="accuracy lower bound the audit must reach for acceptance to be enabled")
    ap.add_argument("--selection-target", type=float, default=None,
                    help="lower bound the threshold must reach on the selection part (default: target + 0.01)")
    args = ap.parse_args()
    if not 0 < args.target < 1:
        raise ValueError("--target must be in (0, 1)")
    if args.selection_target is None:
        args.selection_target = round(args.target + .01, 4)
    settings = Settings.from_env()
    rows = read_rows(args.data)
    from training_frozen_fusion.shared.dataset import load_rows
    old_queries = [r["query"] for split in ("train", "val", "test") for r in load_rows(split)]
    fit_rows, select_rows, audit_rows = partition(rows, old_queries)
    from src.classifier import FusionClassifier
    classifier = FusionClassifier.load(settings.classifier_artifact_path, settings.e5_model_path)
    extractor = FieldExtractor.load(use_tagger=settings.extractor_use_tagger)
    parser = LocalParser(classifier, extractor, ConfidenceGate())
    X, y, _, _, outside_fit = labeled_features(parser, fit_rows)
    if len(set(y)) != 2:
        raise ValueError("confidence fit partition requires both correct and incorrect parses")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=42)).fit(X, y)
    Xs, ys, es, gs, outside_sel = labeled_features(parser, select_rows)
    ps = model.predict_proba(Xs)[:, 1]
    threshold = select_threshold(ps, ys, es, gs, args.selection_target)
    Xa, ya, ea, ga, outside_audit = labeled_features(parser, audit_rows)
    pa = model.predict_proba(Xa)[:, 1]
    audit = operating_point(pa, ya, ea, ga, threshold)
    enabled = threshold is not None and audit["accuracy_lower_95"] >= args.target
    bound = bindings(settings.classifier_artifact_path, settings.e5_model_path, settings.extractor_use_tagger)
    report = {"schema": "parse-confidence-1", "target_accuracy": args.target, "acceptance_enabled": enabled,
              "selection_target": args.selection_target, "threshold": threshold,
              "fit_n": len(fit_rows), "fit_correct_rate": float(y.mean()),
              "groups": {"fit": len({r.group_id for r in fit_rows}), "selection": len(set(gs)),
                         "audit": len(set(ga))},
              "gold_outside_schema": {"fit": outside_fit, "selection": outside_sel, "audit": outside_audit},
              "label_schema": "v2-point-fields",
              "selection": operating_point(ps, ys, es, gs, threshold),
              "audit": audit, "audit_slices": slices(audit_rows, pa, ya, ea, ga, threshold),
              "selection_slices": slices(select_rows, ps, ys, es, gs, threshold),
              "split_seed": 42, "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "partition_ids": {k: [r.id for r in v] for k, v in
                                (("fit", fit_rows), ("selection", select_rows), ("audit", audit_rows))},
              "bindings": bound,
              "limitation": "Wilson bounds count accepted groups (units), not rows. The routing batch is synthetic and LLM-written; "
                            "Reliability applies to the supplied data distribution, not arbitrary production traffic."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**report, "features": FEATURE_NAMES, "model": model}, args.output, compress=3)
    print(f"acceptance_enabled={enabled}; threshold={threshold}; audit={audit}")
    return 0 if enabled else 2


if __name__ == "__main__":
    raise SystemExit(main())
