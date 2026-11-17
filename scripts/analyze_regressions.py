"""Analyze paired regression records without changing prompts, labels, or scoring."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path

from scripts.extraction_rca import analyze
from scripts.evaluate_few_shots import read_records, paired_changes
from src.few_shots import EXAMPLES
from src.openai_catalog import Catalog, VEHICLE_KEYS


def build_analysis(baseline, candidate, catalog=None):
    catalog = catalog or Catalog()
    changes = paired_changes(baseline, candidate)
    regressions = [candidate[key] for key in changes["newly_broken_ids"]]
    details = analyze(regressions)["failures"]
    missing, extra, wrong, domains = Counter(), Counter(), Counter(), Counter()
    global_counts, global_ids = Counter(), set()
    reverse = {value: key for key, value in VEHICLE_KEYS.items()}
    for detail in details:
        detail["baseline_prediction"] = baseline[detail["id"]]["pred"]
        domains[str(detail["gold"]["category"])] += 1
        missing.update(detail["missing_fields"])
        extra.update(detail["extra_fields"])
        wrong.update(detail["wrong_value_fields"])
        gold = detail["gold"]
        internal = {reverse.get(k, k): v for k, v in gold["params"].items()} if gold["category"] == "רכב" else gold["params"]
        global_fields = []
        for name in detail["missing_fields"]:
            entries = [entry for entry in catalog.entries if entry["category"] == gold["category"]
                       and entry["name"] == name and all(internal.get(k) == v for k, v in entry["conditions"].items())]
            if entries and all(not entry["conditions"] for entry in entries):
                global_fields.append(name)
                global_counts[name] += 1
                global_ids.add(detail["id"])
        detail["missing_fields_available_globally"] = global_fields
        if "דגם" in detail["missing_fields"]:
            entries = [entry for entry in catalog.entries if entry["category"] == "רכב" and entry["name"] == "דגמים"
                       and all(internal.get(k) == v for k, v in entry["conditions"].items())]
            detail["omitted_model_is_in_catalog"] = any(gold["params"]["דגם"] in entry["values"] for entry in entries)
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))["examples"]
    return {"total_queries": len(candidate), "newly_wrong": len(details),
            "corrected": len(changes["corrected_ids"]),
            "exclusive_groups": dict(Counter(detail["primary_failure"] for detail in details)),
            "by_category": dict(domains), "missing_fields": dict(missing), "extra_fields": dict(extra),
            "wrong_value_fields": dict(wrong), "global_field_omission_counts_overlapping": dict(global_counts),
            "queries_with_global_field_omissions": len(global_ids),
            "example_field_coverage": dict(Counter(name for example in examples for name in example["expected"]["params"])),
            "limitations": ["Newly wrong means a paired outcome change, not proof of causation by examples.",
                            "One model sample was collected per prompt/query; sampling variability is not estimated.",
                            "Global-field and model omissions are observable patterns. Their proposed prompt mechanisms require controlled validation.",
                            "Rejected raw model output is unavailable; HTTP 502 causes cannot be diagnosed further from these records.",
                            "The original test labels and scores are unchanged; some labels may be incomplete or ambiguous."],
            "findings": [
                "Global inheritance: color and condition can remain valid even when absent from a selected subcategory's own field list.",
                "Demonstration coverage: no demonstration includes color; only one includes condition, and it has no specific item hierarchy.",
                "Vehicle coverage: only one demonstration includes a model. Listed models are being omitted on some previously correct queries.",
                "Over-inference persists: generic second-hand wording sometimes adds used condition despite the intended omission contrast.",
                "At least syn_1919 explicitly requests used condition but its label omits that field; its regression is a label conflict."],
            "next_experiment": [
                "Keep catalog and scoring fixed. Add an explicit field-inheritance instruction and coverage check for stated constraints.",
                "Replace weaker demonstrations with reviewed training contrasts: TV/laptop color and condition, and manufacturer plus named model.",
                "Clarify that demonstrations are illustrative, not exhaustive lists of applicable fields; preserve explicit constraints while avoiding inferred defaults.",
                "Compare proposed variants on validation data; repeat matched 0/6/12-shot cases to separate systematic changes from sampling noise.",
                "Track newly broken cases and category metrics as well as total accuracy; do not teach ambiguous test labels."],
            "regressions": details}


def render(report):
    text = lambda value: escape(str(value), quote=True)
    sections = []
    for title, key in [("Exclusive failure groups", "exclusive_groups"), ("Categories", "by_category"),
                       ("Missing fields", "missing_fields"), ("Extra fields", "extra_fields")]:
        sections += [f"<h2>{title}</h2><table><tr><th>Group or field</th><th>Count</th></tr>"]
        sections += [f'<tr><td dir="auto">{text(name)}</td><td>{count}</td></tr>' for name, count in sorted(report[key].items(), key=lambda pair: -pair[1])]
        sections += ["</table>"]
    for title, key in [("Evidence and likely mechanisms", "findings"), ("Limits of this analysis", "limitations"), ("Recommended controlled experiment", "next_experiment")]:
        sections += [f"<h2>{title}</h2><ul>" + ''.join(f"<li>{text(item)}</li>" for item in report[key]) + "</ul>"]
    rows = []
    for detail in report["regressions"]:
        rows.append(f'<tr><td>{text(detail["id"])}</td><td dir="auto">{text(detail["query"])}</td>'
                    f'<td>{text(detail["primary_failure"])}</td><td><details><summary>Before, after and evidence</summary>'
                    f'<pre>{text(json.dumps(detail, ensure_ascii=False, indent=2))}</pre></details></td></tr>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>New regression analysis</title><style>body{font:15px/1.6 system-ui;margin:32px;color:#253328;background:#fafbf8}'
            'table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #d7ded5;padding:10px;text-align:left;vertical-align:top}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:680px}input{padding:12px;width:80%}</style>'
            '<h1>Queries that became incorrect</h1>'
            f'<p>{report["newly_wrong"]} newly wrong queries, versus {report["corrected"]} corrected queries, out of {report["total_queries"]}.</p>'
            f'<p>{report["queries_with_global_field_omissions"]} affected queries omit globally available fields. Pattern counts can overlap.</p>'
            '<p><a href="regressions.json">JSON evidence</a> · <a href="REPORT.html">Full comparison</a></p>'
            + ''.join(sections) + '<h2>All newly incorrect queries</h2><input id="search" aria-label="Search regressions" placeholder="Search ID, query or evidence">'
            '<table><thead><tr><th>ID</th><th>Query</th><th>Failure</th><th>Evidence</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table><script>document.getElementById("search").addEventListener("input",e=>{'
            'for(const row of document.querySelectorAll("tbody tr"))row.hidden=!row.textContent.toLowerCase().includes(e.target.value.toLowerCase())})</script></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    args = parser.parse_args()
    saved = json.loads((args.comparison / "comparison.json").read_text(encoding="utf-8"))
    candidate = read_records(saved["test_regression"]["directory"])
    baseline = read_records(Path(__file__).resolve().parents[1] / "eval/results/api_test_20261117T085502Z_d07dc252")
    report = build_analysis(baseline, candidate)
    (args.comparison / "regressions.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.comparison / "REGRESSIONS.html").write_text(render(report), encoding="utf-8")
    print(args.comparison / "REGRESSIONS.html")


if __name__ == "__main__":
    main()
