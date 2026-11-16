"""Error analysis of the frozen parser on the routing batch's DEVELOPMENT rows (fit + selection partitions).

The audit partition (20% of units, seed 42 — the same partition scripts/calibrate_confidence.py makes) is sealed: its
rows are split off and never parsed, inspected or summarised. Fit + selection were already inspected during the first calibration, so they become
development data for parser improvements (see README.md for the data-role consequences).

For every dev row it records, per field:
  - classifier: gold field selected or not, predicted extra fields;
  - S1 (gold fields supplied to the extractor): value right or wrong;
  - pipeline: value in the final JSON right / wrong / missing / extra.

python training_frozen_fusion/experiment_5_routing_errors/analyze.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from scripts.calibrate_confidence import partition, read_rows  # noqa: E402
from src.classifier import FusionClassifier  # noqa: E402
from src.extraction.extractor import FieldExtractor  # noqa: E402
from src.extraction.types import values_equal  # noqa: E402
from src.settings import Settings  # noqa: E402
from training_frozen_fusion.shared.dataset import load_rows  # noqa: E402

DATA = REPO / "training_frozen_fusion/data/fresh_routing_calibration/routing.jsonl"
OUT = HERE / "results"


def dev_rows():
    rows = read_rows(DATA)
    fit, selection, audit = partition(rows, [r["query"] for s in ("train", "val", "test") for r in load_rows(s)])
    sealed = {r.id for r in audit}
    dev = [(r, "fit") for r in fit] + [(r, "selection") for r in selection]
    assert not sealed & {r.id for r, _ in dev}
    return dev, len(audit)


def analyse(row, part, classifier, extractor):
    c = classifier.classify(row.query)
    gold = row.expected["params"]
    domain = row.domain
    record = {"id": row.id, "part": part, "query": row.query, "gold_domain": domain, "pred_domain": c.domain,
              "language": row.language, "quality": row.quality, "label_seen_before": row.label_seen_before,
              "gold": gold, "fields": {}}
    pipeline = extractor.extract_all(row.query, c.domain, c.fields)
    payload = extractor.to_json(c.domain, pipeline)
    pred = (payload or {}).get("params") or {}
    record["pred"] = pred
    s1 = extractor.extract_all(row.query, domain, [f"{domain}/{k}" for k in gold]) if domain else {}
    selected = {f.split("/", 1)[1] for f in c.fields if f.startswith(f"{c.domain}/")}
    for name in sorted(set(gold) | set(pred) | selected):
        g, p = gold.get(name), pred.get(name)
        e1 = s1.get(f"{domain}/{name}")
        ep = pipeline.get(f"{c.domain}/{name}")
        if name in gold and name in pred:
            outcome = "right" if values_equal(p, g) else "wrong"
        elif name in gold:
            outcome = "missed_by_classifier" if name not in selected or c.domain != domain else "dropped_no_value"
        elif name in pred:
            outcome = "extra"
        else:
            outcome = "extra_dropped"  # selected, no value found, not in gold: harmless
        record["fields"][name] = {
            "outcome": outcome, "gold": g, "pred": p,
            "s1_value": e1.value if e1 else None, "s1_right": bool(e1 and e1.found and values_equal(e1.value, g)),
            "span": ep.text if ep and ep.found else None, "source": ep.source if ep and ep.found else None}
    exact = domain is not None and c.domain == domain and set(pred) == set(gold) and all(
        values_equal(pred[k], v) for k, v in gold.items())
    record["exact"] = bool(exact)
    return record


def summarise(records):
    out = {"rows": len(records)}
    search = [r for r in records if r["gold_domain"]]
    out["search_rows"] = len(search)
    out["json_exact_search"] = sum(r["exact"] for r in search) / len(search)
    out["domain_correct_search"] = sum(r["gold_domain"] == r["pred_domain"] for r in search) / len(search)
    nonsearch = [r for r in records if not r["gold_domain"]]
    out["nonsearch_rows"] = len(nonsearch)
    out["nonsearch_predicted_search"] = sum(r["pred_domain"] != "none" for r in nonsearch)
    out["nonsearch_with_payload"] = sum(bool(r["pred"]) and r["pred_domain"] != "none" for r in nonsearch)
    outcomes = collections.Counter(f["outcome"] for r in search for f in r["fields"].values())
    out["field_outcomes"] = dict(outcomes)
    gold_fields = [f for r in search for f in r["fields"].values() if f["gold"] is not None]
    out["s1_value_accuracy"] = sum(f["s1_right"] for f in gold_fields) / len(gold_fields)
    # first failing stage per non-exact search row
    stages = collections.Counter()
    for r in search:
        if r["exact"]:
            continue
        o = {f["outcome"] for f in r["fields"].values()}
        if r["gold_domain"] != r["pred_domain"]:
            stages["domain"] += 1
        elif "missed_by_classifier" in o:
            stages["classifier_missed_field"] += 1
        elif "extra" in o:
            stages["extra_field_with_value"] += 1
        elif "dropped_no_value" in o:
            stages["extractor_found_nothing"] += 1
        else:
            stages["wrong_value"] += 1
    out["first_failing_stage"] = dict(stages.most_common())
    per_field = collections.defaultdict(collections.Counter)
    for r in search:
        for name, f in r["fields"].items():
            per_field[f"{r['gold_domain']}/{name}"][f["outcome"]] += 1
    worst = sorted(per_field.items(), key=lambda kv: -(kv[1]["missed_by_classifier"] + kv[1]["wrong"]
                                                          + kv[1]["extra"] + kv[1]["dropped_no_value"]))
    out["fields_by_errors"] = {k: dict(v) for k, v in worst[:30]}
    by = {}
    for key in ("language", "quality", "label_seen_before"):
        groups = collections.defaultdict(list)
        for r in search:
            groups[r[key]].append(r["exact"])
        by[key] = {k: {"n": len(v), "json_exact": round(sum(v) / len(v), 3)} for k, v in sorted(groups.items())}
    out["slices"] = by
    return out


def main():
    settings = Settings.from_env()
    classifier = FusionClassifier.load(settings.classifier_artifact_path, settings.e5_model_path)
    extractor = FieldExtractor.load(use_tagger=settings.extractor_use_tagger)
    dev, sealed_n = dev_rows()
    records = [analyse(r, part, classifier, extractor) for r, part in dev]
    OUT.mkdir(exist_ok=True)
    with open(OUT / "dev_records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    summary = summarise(records)
    summary["sealed_audit_rows_not_loaded"] = sealed_n
    (OUT / "summary_dev.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "fields_by_errors"}, ensure_ascii=False, indent=1))
    for k, v in list(summary["fields_by_errors"].items())[:20]:
        print(k, v)


if __name__ == "__main__":
    main()
