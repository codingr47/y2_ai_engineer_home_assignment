"""Explain saved failures without changing labels, predictions, or scores."""
import argparse
from collections import Counter
from datetime import datetime
from html import escape
import hashlib
import json
from pathlib import Path

from src.extraction.types import values_equal


def same_date(left, right):
    try:
        return datetime.strptime(left, "%Y-%m-%d") == datetime.strptime(right, "%d-%m-%Y")
    except (TypeError, ValueError):
        return False


REVIEWED_LABEL_CONFLICTS = {
    "syn_0501": "The query explicitly says 'אליפטיקל משומש', but the expected params omit מצב. Do not teach omission of an explicit condition.",
    "syn_0621": "The query explicitly says 'משומשת', but the expected params omit מצב. Do not teach omission of an explicit condition.",
    "syn_4694": "An unqualified place name is labeled as a city but predicted as a neighborhood. The query does not establish which geographic role is intended; requires adjudication.",
}

DECISIONS = {
    "vehicle_type_added": "Contrast generic vehicle wording with an explicitly requested vehicle type.",
    "vehicle_year_alias": "Demonstrate exact and bounded model years using שנה; leave catalog metadata unchanged.",
    "date_format_equivalent": "Demonstrate DD-MM-YYYY without changing the scored date string.",
    "entry_date": "Demonstrate absolute and immediate entry dates.",
    "condition": "Contrast generic second-hand marketplace wording with explicitly stated condition; exclude conflicting labels.",
    "fuel": "Demonstrate fuel alternatives separately from vehicle types.",
    "location": "Demonstrate supported canonical city aliases and neighborhood-only extraction; do not memorize ambiguous place labels.",
    "category": "Review ambiguous searches separately; use clear positive and negative intent examples.",
    "other_fields_or_values": "Review query evidence before choosing a demonstration; no automatic corrective rule.",
    "invalid_provider_output": "Unresolved: saved HTTP records do not contain the rejected raw model output.",
}


def analyze(records):
    failures = []
    primary = Counter()
    for row in records:
        if row["json_exact"]:
            continue
        tags, evidence = [], []
        missing, extra, wrong = [], [], []
        if row["error"]:
            group = "request_failed"
            tags.append("invalid_provider_output")
            evidence.append("Saved request error: " + row["error"])
        elif row["gold"]["category"] != row["pred"]["category"]:
            group = "category_wrong"
            tags.append("category")
            evidence.append("Expected and predicted categories differ.")
        else:
            gold, pred = row["gold"]["params"], row["pred"]["params"]
            missing, extra = sorted(set(gold) - set(pred)), sorted(set(pred) - set(gold))
            wrong = sorted(k for k in set(gold) & set(pred) if not values_equal(gold[k], pred[k]))
            group = "field_set_wrong" if missing or extra else "values_only_wrong"
            if "סוגי_רכב" in extra:
                tags.append("vehicle_type_added")
                evidence.append("Vehicle type is present in prediction but absent from expected params; semantic correctness must be reviewed.")
            if "שנות_ייצור" in pred and "שנה" in gold:
                tags.append("vehicle_year_alias")
                evidence.append("Prediction uses שנות_ייצור while normalized labels require שנה.")
            if "תאריך_כניסה" in wrong and same_date(pred["תאריך_כניסה"], gold["תאריך_כניסה"]):
                tags.append("date_format_equivalent")
                evidence.append("Both strings denote the same calendar date; YYYY-MM-DD differs from expected DD-MM-YYYY.")
            affected = set(missing + extra + wrong)
            for field, tag in [("תאריך_כניסה", "entry_date"), ("מצב", "condition"), ("סוג_דלק", "fuel")]:
                if field in affected:
                    tags.append(tag)
            if affected & {"עיר", "שכונה", "רחוב"}:
                tags.append("location")
            if not tags:
                tags.append("other_fields_or_values")
        primary[group] += 1
        review = REVIEWED_LABEL_CONFLICTS.get(row["id"])
        if review:
            evidence.append(review)
        failures.append({"id": row["id"], "query": row["query"], "gold": row["gold"], "pred": row["pred"],
                         "error": row["error"], "primary_failure": group,
                         "missing_fields": missing, "extra_fields": extra, "wrong_value_fields": wrong,
                         "root_cause_tags": tags, "evidence": evidence,
                         "review_status": "label_conflict_or_ambiguity_reviewed" if review else
                                          "format_equivalence_confirmed" if "date_format_equivalent" in tags else
                                          "unresolved_provider_output" if row["error"] else "pattern_identified_semantic_review_needed",
                         "demonstration_eligible": False,
                         "reinforcement_decisions": [DECISIONS[tag] for tag in tags]})
    return {"baseline_rows": len(records), "baseline_correct": sum(row["json_exact"] for row in records),
            "failure_rows": len(failures), "primary_failures": dict(primary),
            "tag_counts_overlapping": dict(Counter(tag for row in failures for tag in row["root_cause_tags"])),
            "method": "Mechanical field/value differences plus reviewed evidence. Tags overlap; unreviewed semantic hypotheses are not confirmed root causes. Test rows are never demonstration candidates.",
            "decisions": DECISIONS, "failures": failures}


def render(report):
    def text(value):
        return escape(str(value), quote=True)
    rows = []
    for failure in report["failures"]:
        detail = json.dumps(failure, ensure_ascii=False, indent=2)
        rows.append(f'<tr><td>{text(failure["id"])}</td><td dir="auto">{text(failure["query"])}</td>'
                    f'<td>{text(", ".join(failure["root_cause_tags"]))}</td><td>{text(failure["review_status"])}</td>'
                    f'<td><details><summary>Evidence and decision</summary><pre>{text(detail)}</pre></details></td></tr>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Extraction root cause analysis</title><style>body{font:15px/1.6 system-ui;margin:32px;color:#253328;background:#fafbf8}'
            'table{border-collapse:collapse;width:100%}th,td{border:1px solid #d7ded5;padding:10px;vertical-align:top;text-align:left}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:600px}input{padding:10px;width:80%;margin:16px 0}</style>'
            '<h1>Extraction root cause analysis</h1>'
            f'<p>{report["failure_rows"]} failures out of {report["baseline_rows"]} queries. Original scores and labels are unchanged.</p>'
            f'<p>{text(report["method"])}</p><h2>Exclusive failure groups</h2><pre>{text(json.dumps(report["primary_failures"], indent=2))}</pre>'
            f'<h2>Overlapping patterns</h2><pre>{text(json.dumps(report["tag_counts_overlapping"], indent=2))}</pre>'
            '<p><a href="RCA.json">JSON evidence</a> · <a href="REPORT.html">Original evaluation</a></p>'
            '<input id="search" type="search" placeholder="Search IDs, queries, causes or evidence" aria-label="Search failure evidence">'
            '<table><thead><tr><th>ID</th><th>Query</th><th>Patterns</th><th>Review status</th><th>Details</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table><script>document.getElementById("search").addEventListener("input",e=>{'
            'for(const row of document.querySelectorAll("tbody tr"))row.hidden=!row.textContent.toLowerCase().includes(e.target.value.toLowerCase())})'
            '</script></html>')


def write_rca(directory):
    directory = Path(directory)
    data = (directory / "records.jsonl").read_bytes()
    report = analyze([json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()])
    report["baseline_records_sha256"] = hashlib.sha256(data).hexdigest()
    (directory / "RCA.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "RCA.html").write_text(render(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = write_rca(args.directory)
    print(json.dumps({k: result[k] for k in ("baseline_rows", "failure_rows", "primary_failures")}))
