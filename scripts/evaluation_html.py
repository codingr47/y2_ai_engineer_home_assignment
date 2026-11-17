"""Standalone, searchable HTML evaluation reports; no external assets required."""
import argparse
from html import escape
import json
from pathlib import Path


def render_html(summary, records):
    from scripts.evaluation_cost import estimate_cost
    cost = summary.get("cost") or estimate_cost(summary)
    def text(value):
        return escape(str(value), quote=True)

    def percent(value):
        return "—" if value is None else f"{value:.1%}"

    def latency(value):
        return "—" if value is None else f"{value * 1000:,.0f} ms"

    def money(value):
        return "Unavailable" if value is None else f"${value:,.6f}"

    def code(value):
        return "<pre>" + text(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"

    config, overall = summary["configuration"], summary["overall"]
    failure_count = sum(overall.get("failures", {}).values())
    cards = [("Test queries", str(overall["rows"])),
             ("Whole JSON correct", percent(overall.get("json_exact"))),
             ("Category correct", percent(overall.get("category_correct"))),
             ("HTTP p95" if config["mode"] == "api" else "Parser p95", latency(overall.get("latency_p95_seconds"))),
             ("Successful calls p95", latency(overall.get("successful_latency_p95_seconds"))),
             ("Failed requests", str(failure_count)),
             ("Estimated run cost", money(cost.get("estimated_total_usd"))),
             ("Estimated cost / query", money(cost.get("estimated_per_query_usd")))]
    chunks = ["""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parser evaluation results</title><style>
:root{color-scheme:light dark;--bg:#fcfcfb;--panel:#fff;--ink:#202320;--muted:#61685f;--line:#dedfd8;--accent:#286a4e}
@media(prefers-color-scheme:dark){:root{--bg:#191c19;--panel:#222822;--ink:#e9eee7;--muted:#acb6a7;--line:#3c453a;--accent:#92cba8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,"Segoe UI",Arial,sans-serif}
main{max-width:1320px;margin:auto;padding:40px 24px}h1{font-size:32px;margin:0}h2{font-size:22px;margin-top:38px}
p{color:var(--muted)}a{color:var(--accent)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:26px 0}
.card{padding:18px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}.card span{display:block;color:var(--muted);font-size:13px}
.card strong{font-size:28px;font-weight:650}.table{overflow:auto;border:1px solid var(--line);border-radius:8px;margin:14px 0}
table{width:100%;border-collapse:collapse;background:var(--panel)}th,td{text-align:left;padding:12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);white-space:nowrap}td.query{min-width:180px;max-width:420px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 Consolas,monospace;unicode-bidi:plaintext}
details summary{cursor:pointer;color:var(--accent)}.toolbar{display:flex;gap:12px;flex-wrap:wrap}input,select{font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}input{flex:1;min-width:180px}
.badge{font-size:12px;font-weight:600}.correct{color:var(--accent)}.incorrect{color:#bb7b16}.failed{color:#cf5252}[hidden]{display:none!important}
</style></head><body><main><h1>Parser evaluation results</h1>"""]
    chunks += [f"<p>{text(config.get('model') or 'Local parser')} · reasoning: {text(config.get('reasoning_effort') or 'model default')} · {text(config['split'])} split · {text(config['mode'])} evaluation · {text(config.get('few_shot_count', 0))} shots</p>",
               '<div class="cards">']
    chunks += [f'<div class="card"><span>{text(label)}</span><strong>{text(value)}</strong></div>' for label, value in cards]
    chunks += ['</div>', f"<p>{text(config.get('latency_scope', 'Direct parser invocation; HTTP is not included.'))}</p>",
               "<p>Failed requests count as incorrect and are included in overall p95. Model confidence is an uncalibrated self-rating.</p>",
               '<p><a href="summary.json">JSON summary</a> · <a href="records.jsonl">Per-query JSONL</a> · <a href="REPORT.md">Markdown report</a></p>',
               '<h2>Configuration and deployment</h2><details><summary>Dataset, model, Docker image, cache, and runtime parameters</summary>',
               code({"configuration": config, "deployment": summary.get("deployment"), "cache": summary.get("cache"),
                     "routing": summary.get("routing"), "provider_outcomes": summary.get("provider_outcomes"),
                     "tokens": overall.get("tokens"), "failures": overall.get("failures"), "http_statuses": summary.get("http_statuses")}), '</details>']
    if config.get("evaluation_role") == "exposed_regression":
        chunks += ['<p><strong>Exposed regression test:</strong> Earlier test failures informed the RCA. This rerun is not an independent accuracy claim.</p>']
    chunks += ['<h2>Estimated OpenAI cost</h2>',
               f'<p>USD · Standard processing · prices verified {text(cost["verified_on"])}. Based on reported usage; this is an estimate, not an invoice.</p>']
    if summary.get("request_cost_coverage"):
        coverage = summary["request_cost_coverage"]
        chunks += [f'<p>Individual request costs available: {coverage["priced_requests"]} of {overall["rows"]}. '
                   f'Sum of priced requests: {money(coverage["sum_estimated_cost_usd"])}.</p>']
    if cost["status"] == "estimated":
        chunks += [f'<p>Projected cost at 10 million queries: <strong>{money(cost["projected_10m_queries_usd"])}</strong>, assuming the same query mix, routing, token usage, and cache behavior.</p>',
                   '<div class="table"><table><thead><tr><th>Token type</th><th>Billable tokens</th><th>USD / 1M tokens</th><th>Estimated cost</th></tr></thead><tbody>']
        for kind, count in cost["billable_tokens"].items():
            chunks.append(f'<tr><td>{text(kind)}</td><td>{count:,}</td><td>{money(cost["rates_per_million_tokens"][kind])}</td><td>{money(cost["breakdown_usd"][kind])}</td></tr>')
        chunks += ['</tbody></table></div>']
    else:
        chunks += [f'<p>Estimate unavailable: {text(cost["reason"])}.</p>']
    chunks += ['<ul>' + ''.join(f'<li>{text(note)}</li>' for note in cost["assumptions"]) + '</ul>',
               f'<p><a href="{text(cost["source"])}">Official model pricing source</a> · Price snapshot: {text(cost["price_version"])}</p>']
    for key, groups in summary.get("slices", {}).items():
        chunks += [f"<h2>By {text(key)}</h2>", '<div class="table"><table><thead><tr><th>Slice</th><th>Queries</th><th>Category correct</th><th>Field set exact</th><th>Whole JSON correct</th><th>p95</th></tr></thead><tbody>']
        for name, group in groups.items():
            values = [name, group["rows"], percent(group.get("category_correct")), percent(group.get("field_set_exact")),
                      percent(group.get("json_exact")), latency(group.get("latency_p95_seconds"))]
            chunks.append("<tr>" + "".join(f'<td dir="auto">{text(value)}</td>' for value in values) + "</tr>")
        chunks += ["</tbody></table></div>"]
    chunks += ['<h2>Query results</h2><div class="toolbar"><input id="search" type="search" aria-label="Search queries" placeholder="Search query text, ID, or extracted values">',
               '<select id="outcome" aria-label="Filter result"><option value="all">All results</option><option value="incorrect">Incorrect parses</option><option value="failed">Failed requests</option><option value="correct">Correct parses</option></select></div>',
               '<p id="count" aria-live="polite"></p><div class="table"><table id="results"><thead><tr><th>ID</th><th>Query</th><th>Result</th><th>HTTP</th><th>Latency</th><th>Estimated cost</th><th>Gold and prediction</th></tr></thead><tbody>']
    for row in records:
        outcome = "failed" if row.get("error") else "correct" if row["json_exact"] else "incorrect"
        chunks += [f'<tr data-outcome="{outcome}"><td>{text(row["id"])}</td><td class="query" dir="auto">{text(row["query"])}</td>',
                   f'<td><span class="badge {outcome}">{outcome}</span></td><td>{text(row.get("http_status") or "—")}</td><td>{latency(row["latency_seconds"])}</td>',
                   f'<td>{money(row.get("estimated_cost_usd"))}</td>',
                   '<td><details><summary>Compare</summary>', code({"gold": row.get("gold"), "prediction": row.get("pred"), "error": row.get("error"),
                       "usage": row.get("usage"), "cost_status": row.get("cost_status"), "cost_breakdown_usd": row.get("cost_breakdown_usd")}),
                   '</details></td></tr>']
    chunks += ['''</tbody></table></div></main><script>
const rows = [...document.querySelectorAll('#results tbody tr')];
const search = document.getElementById('search'), outcome = document.getElementById('outcome');
function filter() {
 const term = search.value.toLocaleLowerCase(); let visible = 0;
 for (const row of rows) {
  row.hidden = !(outcome.value === 'all' || row.dataset.outcome === outcome.value) || !row.textContent.toLocaleLowerCase().includes(term);
  if (!row.hidden) visible++;
 }
 document.getElementById('count').textContent = `${visible} of ${rows.length} results`;
}
search.addEventListener('input', filter); outcome.addEventListener('change', filter); filter();
</script></body></html>''']
    return "\n".join(chunks)


def regenerate(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    from scripts.evaluation_cost import estimate_cost
    if "cost" not in summary:
        summary["cost"] = estimate_cost(summary)
        (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    path = directory / "REPORT.html"
    path.write_text(render_html(summary, records), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="One run directory, or a parent containing run directories")
    directory = parser.parse_args().directory
    directories = [directory] if (directory / "summary.json").is_file() else sorted(path.parent for path in directory.glob("*/summary.json"))
    for run in directories:
        print(regenerate(run))


if __name__ == "__main__":
    main()
