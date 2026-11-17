"""Compare saved HTTP evaluations on identical data, including prompt guard effects."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path

from scripts.evaluate_few_shots import read_records, paired_changes


def omitted_fields(records):
    counts = Counter()
    for row in records.values():
        if row["pred"] is not None and row["pred"]["category"] == row["gold"]["category"]:
            counts.update(set(row["gold"]["params"]) - set(row["pred"]["params"]))
    return dict(counts)


def compare(baseline_directory, candidate_directory, *, allow_reasoning_change=False, allow_model_change=False):
    if allow_reasoning_change and allow_model_change:
        raise ValueError("change only one comparison axis at a time")
    baseline_directory, candidate_directory = Path(baseline_directory).resolve(), Path(candidate_directory).resolve()
    before = json.loads((baseline_directory / "summary.json").read_text(encoding="utf-8"))
    after = json.loads((candidate_directory / "summary.json").read_text(encoding="utf-8"))
    for key in ("model", "reasoning_effort", "concurrency", "few_shot_count", "examples_sha256", "taxonomy_sha256", "dataset_sha256"):
        if key == "reasoning_effort" and allow_reasoning_change:
            continue
        if key == "model" and allow_model_change:
            continue
        if before["configuration"].get(key) != after["configuration"].get(key):
            raise ValueError("evaluations differ in controlled configuration: " + key)
    if (allow_reasoning_change or allow_model_change) and before["configuration"].get("prompt_sha256") != after["configuration"].get("prompt_sha256"):
        raise ValueError("model/reasoning comparisons require identical prompts")
    if before.get("deployment", {}).get("cache_capacity") != after.get("deployment", {}).get("cache_capacity"):
        raise ValueError("evaluations differ in application cache capacity")
    old_records, new_records = read_records(baseline_directory), read_records(candidate_directory)
    changes = paired_changes(old_records, new_records)
    old, new = before["overall"], after["overall"]
    report = {"comparison_axis": "model" if allow_model_change else "reasoning_effort" if allow_reasoning_change else "guidelines",
              "baseline": {"directory": str(baseline_directory), "summary": before},
              "candidate": {"directory": str(candidate_directory), "summary": after},
              "evaluation_role": "exposed_regression", "changes": changes,
              "accuracy_change_percentage_points": (new["json_exact"] - old["json_exact"]) * 100,
              "p95_change_seconds": new["latency_p95_seconds"] - old["latency_p95_seconds"],
              "category_changes": {name: {"before": group["json_exact"], "after": after["slices"]["category"][name]["json_exact"]}
                                   for name, group in before["slices"]["category"].items()},
              "missing_fields_before": omitted_fields(old_records), "missing_fields_after": omitted_fields(new_records),
              "changed_queries": [{"id": key, "query": new_records[key]["query"], "gold": new_records[key]["gold"],
                                   "before": old_records[key]["pred"], "after": new_records[key]["pred"],
                                   "before_error": old_records[key]["error"], "after_error": new_records[key]["error"],
                                   "outcome": "corrected" if new_records[key]["json_exact"] else "newly_wrong"}
                                  for key in changes["corrected_ids"] + changes["newly_broken_ids"]],
              "limitations": "One run per prompt. Paired differences do not establish causation or measure sampling variability. The test set is exposed."}
    # Look back at the previous experiment's newly broken cases, when available.
    previous = baseline_directory.parent / "comparison.json"
    if previous.is_file():
        previous_report = json.loads(previous.read_text(encoding="utf-8"))
        previous_regression = previous_report.get("test_regression") or {}
        ids = previous_regression.get("changes_from_original_test", {}).get("newly_broken_ids", [])
        if ids:
            report["previous_regression_recovery"] = {"previously_broken": len(ids),
                "now_correct_ids": [key for key in ids if new_records[key]["json_exact"]],
                "still_wrong_ids": [key for key in ids if not new_records[key]["json_exact"]]}
    return report


def render(report):
    def text(value):
        return escape(str(value), quote=True)
    old, new = report["baseline"]["summary"], report["candidate"]["summary"]
    reasoning = report.get("comparison_axis") == "reasoning_effort"
    model = report.get("comparison_axis") == "model"
    title = "Model comparison" if model else "Reasoning effort comparison" if reasoning else "IMPORTANT GUIDELINES comparison"
    old_label = old["configuration"]["model"] if model else str(old["configuration"]["reasoning_effort"]) + " reasoning" if reasoning else "Previous 12-shot run"
    new_label = new["configuration"]["model"] if model else str(new["configuration"]["reasoning_effort"]) + " reasoning" if reasoning else "With IMPORTANT GUIDELINES"
    def percentage(summary):
        return f'{summary["overall"]["json_exact"]:.2%}'
    def p95(summary):
        return f'{summary["overall"]["latency_p95_seconds"]:.3f} s'
    def money(summary):
        value = summary.get("cost", {}).get("estimated_total_usd")
        return f'${value:.6f}' if value is not None else "unavailable"
    rows = [("Whole-JSON accuracy", percentage(old), percentage(new)), ("HTTP p95", p95(old), p95(new)),
            ("Failed requests", sum(old["overall"]["failures"].values()), sum(new["overall"]["failures"].values())),
            ("Estimated cost", money(old), money(new))]
    table = ''.join('<tr>' + ''.join(f'<td>{text(cell)}</td>' for cell in row) + '</tr>' for row in rows)
    attribute_rows = ''.join(f'<tr><td dir="auto">{text(name)}</td><td>{report["missing_fields_before"].get(name, 0)}</td>'
                             f'<td>{report["missing_fields_after"].get(name, 0)}</td></tr>' for name in ("צבע", "מצב", "דגם"))
    category_rows = ''.join(f'<tr><td dir="auto">{text(name)}</td><td>{values["before"]:.2%}</td>'
                           f'<td>{values["after"]:.2%}</td></tr>' for name, values in report["category_changes"].items())
    details = []
    for row in report["changed_queries"]:
        details.append(f'<tr><td>{text(row["id"])}</td><td dir="auto">{text(row["query"])}</td><td>{row["outcome"]}</td>'
                       f'<td><details><summary>Compare</summary><pre>{text(json.dumps(row, ensure_ascii=False, indent=2))}</pre></details></td></tr>')
    metadata = {key: report[key] for key in ["category_changes", "missing_fields_before", "missing_fields_after", "limitations"]}
    recovery = report.get("previous_regression_recovery")
    if recovery:
        metadata["previous_regression_recovery"] = recovery
    metadata["prompt_configuration_before"] = old["configuration"]
    metadata["prompt_configuration_after"] = new["configuration"]
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>' + text(title) + '</title><style>body{font:16px/1.6 system-ui;margin:32px;color:#253328;background:#fafbf8}'
            'table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:10px;border:1px solid #d7ded5;text-align:left;vertical-align:top}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:700px}input{padding:12px;width:80%}</style>'
            '<h1>' + text(title) + '</h1><p>Same model, demonstrations, dataset, HTTP concurrency, and disabled application cache. Compared setting: ' + text(report.get("comparison_axis", "guidelines")) + '.</p>'
            '<table><tr><th>Metric</th><th>' + text(old_label) + '</th><th>' + text(new_label) + '</th></tr>' + table + '</table>'
            f'<p>Corrected {len(report["changes"]["corrected_ids"])} queries; newly broke {len(report["changes"]["newly_broken_ids"])}.</p>'
            '<h2>Targeted field omissions</h2><p>Counts on queries with the correct predicted category; failures and category errors are reported separately.</p>'
            '<table><tr><th>Field</th><th>Before</th><th>After</th></tr>' + attribute_rows + '</table>'
            '<h2>Whole-JSON accuracy by category</h2><table><tr><th>Category</th><th>Before</th><th>After</th></tr>' + category_rows + '</table>'
            '<p><a href="REPORT.html">Full candidate report</a> · <a href="comparison.json">Comparison JSON</a></p>'
            '<p>This is an exposed regression comparison. A single run does not quantify sampling variability.</p>'
            '<details><summary>Configuration, category results, and field omissions</summary><pre>'
            + text(json.dumps(metadata, ensure_ascii=False, indent=2)) + '</pre></details>'
            '<h2>Changed query outcomes</h2><input id="search" aria-label="Search changed outcomes" placeholder="Search ID, query or result">'
            '<table id="changes"><thead><tr><th>ID</th><th>Query</th><th>Change</th><th>Gold and predictions</th></tr></thead><tbody>'
            + ''.join(details) + '</tbody></table><script>document.getElementById("search").addEventListener("input",e=>{'
            'for(const row of document.querySelectorAll("#changes tbody tr"))row.hidden=!row.textContent.toLowerCase().includes(e.target.value.toLowerCase())})</script></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    axis = parser.add_mutually_exclusive_group()
    axis.add_argument("--allow-reasoning-change", action="store_true")
    axis.add_argument("--allow-model-change", action="store_true")
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, allow_reasoning_change=args.allow_reasoning_change,
                     allow_model_change=args.allow_model_change)
    (args.candidate / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.candidate / "COMPARISON.html").write_text(render(report), encoding="utf-8")
    print(args.candidate / "COMPARISON.html")


if __name__ == "__main__":
    main()
