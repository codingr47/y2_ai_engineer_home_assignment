"""End-to-end evaluation of the shipped local parser on the held-out test split (eval/dataset/data/test.jsonl).

Opt-in, because it needs the provisioned encoder (models/multilingual-e5-small) and takes about a minute:

    RUN_EVAL=1 pytest tests/eval/test_parser_on_test_split.py -s

It scores the parser as shipped: the classifier, the extractor and the confidence gate in src/artifacts. Nothing is
tuned here. Test is a final check, and its numbers must never feed back into thresholds or model choices. It
reports:
  - pipeline: domain accuracy, field-set exact and JSON exact over all rows (a non-search is right when nothing is
    parsed);
  - gate: coverage and accuracy of the parses the gate accepts, against the bound the audit certified.

The report is written to eval/results/parser_test_eval.json, per-row records to parser_test_records.jsonl.
"""
import collections
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TEST = REPO / "eval" / "dataset" / "data" / "test.jsonl"
REPORT = REPO / "eval" / "results" / "parser_test_eval.json"
RECORDS = REPO / "eval" / "results" / "parser_test_records.jsonl"  # per row, for error review
DOMAINS = ("נדל״ן", "רכב", "יד_שנייה")

pytestmark = pytest.mark.skipif(os.environ.get("RUN_EVAL") != "1", reason="set RUN_EVAL=1 to run the test-split eval")


def gold_rows():
    from training.training_frozen_fusion.shared.normalize import normalize_row  # labels in schema v2 at load time; the file is untouched
    rows = []
    for line in TEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            rows.append(normalize_row({**raw, "domain": raw["expected"]["category"]}))
    return rows


def wilson_lower(k, n):
    from scripts.calibrate_confidence import wilson_lower as w
    return w(k, n)


@pytest.fixture(scope="module")
def evaluation():
    from src.extraction.types import values_equal
    from src.frozen_fusion.local_parser import LocalParser, public_payload
    from src.config.settings import Settings

    settings = Settings(extractor_enabled=True)
    if not settings.e5_model_path.is_dir():
        pytest.skip("encoder not provisioned at models/multilingual-e5-small (see docs/optional-parser.md)")
    parser = LocalParser.load(settings)
    gate = parser.gate
    records = []
    for row in gold_rows():
        domain = row["domain"]
        gold = public_payload({"category": domain, "params": row["expected"]["params"]})["params"] if domain else None
        cand = parser.candidate(row["query"])
        score = gate.score(cand.vector)
        pred = public_payload(cand.payload) if cand.payload else None
        c = parser.classifier.classify(row["query"])
        if domain is None:
            exact = pred is None or not pred["params"]
            field_set = c.domain == "none"
        else:
            exact = pred is not None and pred["category"] == domain and set(pred["params"]) == set(gold) and all(
                values_equal(pred["params"][k], v) for k, v in gold.items())
            field_set = c.domain == domain and {f.split("/", 1)[1] for f in c.fields} == set(row["expected"]["params"])
        accepted = cand.reason is None and gate.accepts(score)
        records.append({"id": row["id"], "query": row["query"], "gold": gold, "pred": pred and pred["params"],
                        "pred_category": pred and pred["category"], "score": score, "reason": cand.reason,
                        "domain": domain, "pred_domain": c.domain, "exact": bool(exact),
                        "field_set": bool(field_set), "accepted": accepted,
                        "accepted_correct": accepted and domain is not None and bool(exact),
                        "language": "+".join(row["metadata"].get("language") or ["unknown"]),
                        "quality": "gold" if "quality:gold" in (row.get("notes") or "") else "silver"})
    return gate, records


def summarise(gate, records):
    def block(rs):
        n = len(rs)
        acc = [r for r in rs if r["accepted"]]
        k = sum(r["accepted_correct"] for r in acc)
        return {"n": n, "domain_accuracy": round(sum((r["pred_domain"] == (r["domain"] or "none")) for r in rs) / n, 4),
                "field_set_exact": round(sum(r["field_set"] for r in rs) / n, 4),
                "json_exact": round(sum(r["exact"] for r in rs) / n, 4),
                "gate_coverage": round(len(acc) / n, 4),
                "gate_accepted": len(acc),
                "gate_accuracy": round(k / len(acc), 4) if acc else None,
                "gate_accuracy_lower_95": round(wilson_lower(k, len(acc)), 4) if acc else None}

    out = {"split": "test", "rows": len(records), "overall": block(records),
           "search_rows": block([r for r in records if r["domain"]]),
           "gate": {"enabled": gate.enabled, "threshold": gate.threshold,
                    "certified_lower_bound": (gate.artifact or {}).get("certified_lower_bound"),
                    "audit": {k: (gate.artifact or {}).get("audit", {}).get(k) for k in ("coverage", "accuracy")}},
           "slices": {}}
    for key in ("domain", "language", "quality"):
        groups = collections.defaultdict(list)
        for r in records:
            groups[str(r[key])].append(r)
        out["slices"][key] = {k: block(v) for k, v in sorted(groups.items())}
    return out


def test_parser_on_test_split(evaluation):
    gate, records = evaluation
    summary = summarise(gate, records)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    RECORDS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    o = summary["overall"]
    print(f"\ntest ({o['n']} rows): domain {o['domain_accuracy']:.1%}, field set {o['field_set_exact']:.1%}, "
          f"JSON exact {o['json_exact']:.1%}; gate accepts {o['gate_coverage']:.1%} at {o['gate_accuracy']} "
          f"(lower bound {o['gate_accuracy_lower_95']}); report {REPORT.relative_to(REPO)}")

    assert summary["rows"] == 2181
    assert o["domain_accuracy"] > .9  # sanity: the shipped classifier is the one we think it is
    if gate.enabled:
        # The audit's claim must hold on independent data: accepted parses at least as accurate as the certified bound.
        assert o["gate_accepted"] > 0
        assert o["gate_accuracy"] >= summary["gate"]["certified_lower_bound"], summary["gate"]
