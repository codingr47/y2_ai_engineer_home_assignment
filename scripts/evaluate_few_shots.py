"""Review provenance, compare 0/6/12 shots on validation, then regress the winner on test."""
import argparse
import asyncio
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from uuid import uuid4

from scripts.evaluate_docker import run_docker_evaluation
from scripts.evaluate_parser import load_rows
from src.few_shots import EXAMPLES, prompt_configuration
from src.openai_catalog import Catalog
from src.settings import ROOT, Settings


def normalized_query(query):
    return re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", query).casefold())


def validate_examples(*, guidelines_enabled=False):
    from training_frozen_fusion.shared.dataset import load_rows as training_rows
    artifact = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    train = {row["id"]: row for row in training_rows("train")}
    held_out = training_rows("val") + training_rows("test") + load_rows("test")
    held_ids = {row["id"] for row in held_out}
    held_queries = {normalized_query(row["query"]) for row in held_out}
    held_families = {row.get("family") for row in held_out if row.get("family")}
    seen_ids, seen_families = set(), set()
    for example in artifact["examples"]:
        source = train.get(example["source_id"])
        if source is None or example["query"] != source["query"] or example["expected"] != source["expected"]:
            raise ValueError("example must exactly match a normalized training row")
        if (example["source_split"] != "train" or example["review_status"] != "reviewed"
                or not example["review_note"] or example["family"] != source["family"]):
            raise ValueError("example review or provenance is incomplete")
        if (example["source_id"] in held_ids or normalized_query(example["query"]) in held_queries
                or example["family"] in held_families):
            raise ValueError("few-shot example overlaps validation or test data")
        if example["source_id"] in seen_ids or example["family"] in seen_families:
            raise ValueError("few-shot examples must use distinct IDs and families")
        seen_ids.add(example["source_id"])
        seen_families.add(example["family"])
    catalog = Catalog()
    configurations = {str(count): prompt_configuration(catalog, count, guidelines_enabled=guidelines_enabled)[1]
                      for count in (0, 6, 12)}
    return {"examples": len(artifact["examples"]), "train_rows": len(train),
            "checks": ["source IDs", "source labels", "review notes", "normalized query overlap", "family overlap", "taxonomy validation"],
            "configurations": configurations, "source": str(EXAMPLES.relative_to(ROOT))}


def read_records(directory):
    return {row["id"]: row for line in (Path(directory) / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() for row in [json.loads(line)]}


def paired_changes(baseline, candidate):
    if set(baseline) != set(candidate) or any(baseline[key]["gold"] != candidate[key]["gold"] or
                                           baseline[key]["query"] != candidate[key]["query"] for key in baseline):
        raise ValueError("paired comparison requires identical query IDs, text, and labels")
    return {"corrected_ids": [key for key in baseline if not baseline[key]["json_exact"] and candidate[key]["json_exact"]],
            "newly_broken_ids": [key for key in baseline if baseline[key]["json_exact"] and not candidate[key]["json_exact"]]}


def select_winner(runs):
    return min(runs, key=lambda run: (-run["summary"]["overall"]["json_exact"], run["few_shot_count"],
                                     run["summary"]["overall"]["latency_p95_seconds"]))


def mark_exposed_regression(directory, summary):
    """Label the saved standalone report as well as the comparison manifest."""
    from scripts.evaluation_html import render_html
    directory = Path(directory)
    summary["configuration"]["evaluation_role"] = "exposed_regression"
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    records = list(read_records(directory).values())
    (directory / "REPORT.html").write_text(render_html(summary, records), encoding="utf-8")
    markdown = directory / "REPORT.md"
    note = "Evaluation role: exposed regression; test failures informed RCA, so this is not an independent accuracy claim."
    content = markdown.read_text(encoding="utf-8")
    if note not in content:
        markdown.write_text(content + "\n" + note + "\n", encoding="utf-8")


def write_comparison(directory, report):
    directory = Path(directory).resolve()
    (directory / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for run in report["validation_runs"]:
        summary, changes = run["summary"], run.get("changes_from_zero", {})
        overall = summary["overall"]
        link = Path(run["directory"]).relative_to(directory).as_posix() + "/REPORT.html"
        cost = summary.get("cost", {}).get("estimated_total_usd")
        cells = [str(run["few_shot_count"]), f'{overall["json_exact"]:.2%}', f'{overall["latency_p95_seconds"]:.3f} s',
                 str(sum(overall["failures"].values())), f'${cost:.6f}' if cost is not None else "unavailable",
                 str(len(changes.get("corrected_ids", []))), str(len(changes.get("newly_broken_ids", [])))]
        rows.append('<tr>' + ''.join(f'<td>{escape(cell)}</td>' for cell in cells)
                    + f'<td><a href="{escape(link, quote=True)}">Full report</a></td></tr>')
    winner = report.get("selected_count")
    regression = report.get("test_regression")
    regression_html = ""
    if regression:
        link = Path(regression["directory"]).relative_to(directory).as_posix() + "/REPORT.html"
        regression_html = f'<p>Selected prompt test result: <a href="{escape(link, quote=True)}">Exposed regression report</a>.</p>'
        if regression.get("baseline_summary"):
            old, new = regression["baseline_summary"], regression["summary"]
            changes = regression["changes_from_original_test"]
            regression_html += (f'<p>Test whole-JSON accuracy: <strong>{old["overall"]["json_exact"]:.2%} → {new["overall"]["json_exact"]:.2%}</strong>. '
                                f'Corrected {len(changes["corrected_ids"])} queries; newly broke {len(changes["newly_broken_ids"])}.</p>'
                                '<table><thead><tr><th>Test category</th><th>Original</th><th>Selected prompt</th><th>Change</th></tr></thead><tbody>')
            for name, group in new["slices"]["category"].items():
                before = old["slices"]["category"][name]["json_exact"]
                after = group["json_exact"]
                regression_html += (f'<tr><td>{escape(name)}</td><td>{before:.2%}</td><td>{after:.2%}</td>'
                                    f'<td>{(after-before)*100:+.2f} percentage points</td></tr>')
            regression_html += '</tbody></table>'
    details = json.dumps({"settings": report["settings"], "selection_rule": report["selection_rule"],
                          "selected_count": winner, "category_deltas": report.get("category_deltas"),
                          "provenance": report["provenance"], "test_regression": regression}, ensure_ascii=False, indent=2)
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Few-shot comparison</title><style>body{font:16px/1.6 system-ui;margin:36px;color:#233c2d;background:#fafbf8}'
            'td,th{padding:12px;border:1px solid #d7ded5;text-align:left}table{border-collapse:collapse}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
            '<h1>Few-shot validation comparison</h1><p>Same model, reasoning effort, validation rows, HTTP concurrency, and disabled application cache.</p>'
            '<p>Selection uses validation whole-JSON accuracy including failures, then fewer examples, then lower p95. Test results do not select the winner.</p>'
            f'<p>Selected example count: <strong>{winner if winner is not None else "Pending"}</strong>.</p>'
            '<table><thead><tr><th>Shots</th><th>JSON exact</th><th>HTTP p95</th><th>Failures</th><th>Estimated cost</th><th>Corrected</th><th>Newly broken</th><th>Results</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>' + regression_html
            + '<p>The test set informed RCA and is exposed. Its rerun is a regression comparison, not an independent accuracy claim. Production defaults and original labels are unchanged.</p>'
            + '<details><summary>Configuration, category changes, and provenance</summary><pre>' + escape(details) + '</pre></details></html>')
    (directory / "REPORT.html").write_text(html, encoding="utf-8")


async def run_comparison(*, settings=None, output=None, allow_paid_calls=False, concurrency=8):
    if not allow_paid_calls:
        raise ValueError("comparison requires explicit paid-call opt-in")
    settings = settings or Settings.from_env()
    provenance = validate_examples(guidelines_enabled=settings.openai_guidelines_enabled)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output or ROOT / "eval/results") / f"few_shots_{stamp}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    report = {"settings": {"model": settings.openai_model, "reasoning_effort": settings.openai_reasoning_effort,
                            "guidelines_enabled": settings.openai_guidelines_enabled,
                            "concurrency": concurrency, "cache_capacity": 0},
              "selection_rule": "Highest validation JSON exact, then fewer examples, then lower p95; failures count as incorrect.",
              "provenance": provenance, "validation_runs": [], "test_regression": None}
    write_comparison(directory, report)
    print(f"Comparison HTML: {directory / 'REPORT.html'}", flush=True)
    for count in (0, 6, 12):
        configured = Settings.model_validate({**settings.model_dump(), "openai_few_shot_count": count})
        print(f"Starting {count}-shot validation evaluation", flush=True)
        run_dir, summary = await run_docker_evaluation(mode="openai", split="val", settings=configured,
                                                       output=directory, allow_paid_calls=True, concurrency=concurrency)
        run = {"few_shot_count": count, "directory": str(run_dir), "summary": summary}
        if report["validation_runs"]:
            run["changes_from_zero"] = paired_changes(read_records(report["validation_runs"][0]["directory"]), read_records(run_dir))
        report["validation_runs"].append(run)
        write_comparison(directory, report)
    winner = select_winner(report["validation_runs"])
    report["selected_count"] = winner["few_shot_count"]
    baseline = report["validation_runs"][0]["summary"]["slices"]["category"]
    report["category_deltas"] = {str(run["few_shot_count"]): {name: group["json_exact"] - baseline[name]["json_exact"]
                                for name, group in run["summary"]["slices"]["category"].items()}
                                 for run in report["validation_runs"]}
    write_comparison(directory, report)
    print(f"Validation selected {winner['few_shot_count']} shots; running exposed test regression", flush=True)
    configured = Settings.model_validate({**settings.model_dump(), "openai_few_shot_count": winner["few_shot_count"]})
    run_dir, summary = await run_docker_evaluation(mode="openai", split="test", settings=configured,
                                                   output=directory, allow_paid_calls=True, concurrency=concurrency)
    mark_exposed_regression(run_dir, summary)
    original = ROOT / "eval/results/api_test_20261117T085502Z_d07dc252"
    regression = {"directory": str(run_dir), "summary": summary, "evaluation_role": "exposed_regression"}
    if original.is_dir():
        regression["changes_from_original_test"] = paired_changes(read_records(original), read_records(run_dir))
        regression["baseline_summary"] = json.loads((original / "summary.json").read_text(encoding="utf-8"))
    report["test_regression"] = regression
    write_comparison(directory, report)
    return directory, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args())
    directory, report = asyncio.run(run_comparison(**args))
    print(f"Comparison HTML: {directory / 'REPORT.html'}; selected shots={report['selected_count']}", flush=True)


if __name__ == "__main__":
    main()
