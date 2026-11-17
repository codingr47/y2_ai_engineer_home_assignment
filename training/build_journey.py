"""Build the offline training narrative from saved results: python -m training.build_journey."""
from __future__ import annotations

import base64
from html import escape
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FF = "training_frozen_fusion"
BERT = "training_bert"


def read(path):
    return json.loads((HERE / path).read_text(encoding="utf-8"))


def pct(value):
    return f"{100 * value:.1f}%"


def source(path, label="Source report"):
    assert (HERE / path).is_file(), path
    return f'<a href="{escape(path)}">{escape(label)} <span aria-hidden="true">↗</span></a>'


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        f'<th scope="col">{escape(str(h))}</th>' for h in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(str(c))}</td>' for c in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def bars(title, subtitle, values):
    """HTML bars stay readable on mobile and expose their values without hover."""
    return f'<figure><h3>{escape(title)}</h3><p class="caption">{escape(subtitle)}</p>' + ''.join(
        f'<div class="bar-row"><span>{escape(label)}</span><div class="track"><div class="bar" '
        f'style="width:{v * 100:.3f}%;--bar-color:{color}"></div></div><strong>{pct(v)}</strong></div>'
        for label, v, color in values
    ) + '<figcaption>Bars use a common 0–100% scale.</figcaption></figure>'


def curve(points):
    x = lambda n: 58 + (n / 18000) * 580
    y = lambda v: 252 - (v - .6) / .35 * 205
    svg = '<svg viewBox="0 0 700 310" role="img" aria-labelledby="curve-title curve-desc"><title id="curve-title">DictaBERT learning curve</title><desc id="curve-desc">'
    svg += escape('; '.join(f'{round(n):,} training rows: {pct(v)} JSON exact' for n, v in points)) + '</desc>'
    for v in (.6, .7, .8, .9):
        svg += f'<path d="M58 {y(v)} H660" stroke="#e2e5e2"/><text x="46" y="{y(v)+4}" text-anchor="end">{v*100:.0f}%</text>'
    svg += '<polyline fill="none" stroke="#16746b" stroke-width="3" points="' + ' '.join(f'{x(n)},{y(v)}' for n, v in points) + '"/>'
    for n, v in points:
        svg += f'<circle cx="{x(n)}" cy="{y(v)}" r="5" fill="#16746b"/><text x="{x(n)}" y="{y(v)-14}" text-anchor="middle">{pct(v)}</text><text x="{x(n)}" y="279" text-anchor="middle">{n/1000:.1f}k</text>'
    return svg + '<text x="350" y="305" text-anchor="middle">Training rows · family-grouped subsets</text></svg>'


def archived_figure(path, caption):
    # Embed selected figures: the complete narrative remains portable as one HTML file.
    data = base64.b64encode((HERE / path).read_bytes()).decode('ascii')
    return f'<figure><img loading="lazy" src="data:image/png;base64,{data}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'


def build():
    comparison = read(f'{FF}/comparison/results/comparison.json')
    v2 = read(f'{FF}/comparison/results/comparison_v2.json')
    summary = read(f'{BERT}/experiment1/results/summary.json')
    clean = read(f'{BERT}/experiment2/results/clean_report_with_test.json')
    gate = read(f'{BERT}/experiment2/results/gate_ff_report_clean_with_test.json')
    calibration = read(f'{BERT}/experiment3/results/calibration_report.json')
    audit = read(f'{BERT}/experiment3/audit/audit_result.json')
    extra = read(f'{FF}/experiment_7_extra_train/results/summary_dev.json')
    # Recorded per-run output contains the directly comparable Frozen Fusion baseline.
    final = read(f'{BERT}/experiment1/runs/e11_final_ensemble/final_final.json')
    test = summary['e11_final']['test']
    chosen = gate['gates']['tagger+ff']['0.95']
    certified = calibration['gates']['tagger+ff']['audit']
    assert clean['test']['rows'] == comparison['headline']['test']['exp3']['n'] == 2181
    colors = ['#a9b7b3', '#7d9a94', '#4e8b85', '#16746b']
    reps = [('E5 · original fields', 'exp1'), ('E5 · merged slots', 'exp1-slots'), ('TF-IDF', 'exp2'), ('E5 + TF-IDF', 'exp3')]
    representation_chart = bars('Surface detail changes the result', 'Test · 2,181 queries · original 70-slot exact-set accuracy', [
        (name, comparison['headline']['test'][key]['slot_exact_set'], color)
        for (name, key), color in zip(reps, colors)
    ])
    representation_table = table(['Representation', 'Dev slot exact', 'Test slot exact', 'Test v2 field exact'], [
        [name, pct(comparison['headline']['val_eval'][key]['slot_exact_set']),
         pct(comparison['headline']['test'][key]['slot_exact_set']), pct(v2['headline']['test'][key]['slot_exact_set'])]
        for name, key in reps
    ])
    points = [(sum(p['rows']) / len(p['rows']), p['json_exact']['mean']) for p in summary['e4_learning_curve']]
    json_chart = bars('The end-to-end improvement', 'Same 2,181 test queries · complete JSON exact accuracy', [
        ('Frozen Fusion', .754, '#a9b7b3'), ('Fine-tuned ensemble', test['json_exact'], '#4e8b85'),
        ('+ runtime cleanup', clean['test']['json_exact_clean'], '#16746b')
    ])
    learning_table = table(['Training rows (mean)', 'Dev JSON exact', 'Runs'], [
        [f'{sum(p["rows"])/len(p["rows"]):,.0f}', pct(p['json_exact']['mean']), p['json_exact']['n']]
        for p in summary['e4_learning_curve']
    ])
    lr_table = table(['Encoder learning rate', 'Best dev JSON', 'Selected epoch'], [
        [f'{p["lr"]:.0e}', pct(p['json_exact']), p['epoch']] for p in summary['e2_lr_sweep']
    ])
    gate_table = table(['Gate / evaluation', 'Locally answered', 'Accepted accuracy', '95% lower bound'], [
        ['E1 tagger gate · test', pct(summary['e12_gate']['chosen']['test']['coverage']), pct(summary['e12_gate']['chosen']['test']['accuracy']), '—'],
        ['E2 cleaned tagger only · test', pct(gate['gates']['tagger']['0.95']['test']['coverage']), pct(gate['gates']['tagger']['0.95']['test']['accuracy']), '—'],
        ['E2 cleaned + fusion opinion · test', pct(chosen['test']['coverage']), pct(chosen['test']['accuracy']), '—'],
        ['E2 gate · routing audit', pct(calibration['expt2_val_gate_on_routing_audit']['coverage']), pct(calibration['expt2_val_gate_on_routing_audit']['accuracy']), pct(calibration['expt2_val_gate_on_routing_audit']['accuracy_lower_95'])],
        ['E3 routing-calibrated · routing audit', pct(certified['coverage']), pct(certified['accuracy']), pct(certified['accuracy_lower_95'])]
    ])
    extra_table = table(['Frozen Fusion variant', 'Training rows', 'Dev JSON · search rows', 'Routing-dev JSON · search rows'], [
        [label, n, pct(extra[key]['val_eval']['json_exact']), pct(extra[key]['routing_dev']['json_exact'])]
        for label, n, key in [('Before extra data', '6,171', 'shipped'), ('+ half extra', '7,766', 'extra50_C16'), ('+ all extra · selected', '9,359', 'extra100_C16')]
    ])
    slices = test['slices']
    slice_table = table(['E1 final tagger · test slice', 'Queries', 'JSON exact'], [
        [label, slices[key]['n'], pct(slices[key]['json_exact'])] for label, key in [
            ('Gold', 'quality=gold'), ('Silver', 'quality=silver'), ('Typos', 'language=typo'),
            ('Standard', 'language=standard'), ('Real estate', 'domain=נדל״ן')]
    ])
    html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="The Yad2 search-parser training journey: frozen features, Hebrew fine-tuning, extraction, and confidence routing.">
<title>From frozen features to fine-tuned search · Training journey</title><style>
:root{--ink:#20312e;--muted:#5a6964;--line:#dce3dd;--paper:#f7f8f3;--green:#16746b;--orange:#bb552a}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:90px}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 'Segoe UI',Arial,sans-serif}a{color:#12685f;text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid #bb552a;outline-offset:5px}.wrap{max-width:1100px;margin:auto;padding:0 36px}header{padding:65px 0 42px;border-top:7px solid var(--green)}.eyebrow,.kicker{font-size:12px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:var(--green)}h1{font-size:clamp(38px,5.2vw,64px);line-height:1.08;letter-spacing:-2.3px;max-width:860px;margin:18px 0 25px;font-weight:650}h2{font-size:32px;line-height:1.2;letter-spacing:-.7px;margin:9px 0 18px}h3{font-size:18px;line-height:1.4;margin:0 0 10px}p{margin:0 0 16px}.intro{font-size:19px;color:var(--muted);max-width:780px}.meta{font-size:13px;color:var(--muted)}.hero-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:32px}.stat{padding:21px 24px;background:#fff;border:1px solid var(--line);border-radius:10px}.stat strong{display:block;font-size:35px;line-height:1.2;letter-spacing:-1px}.stat span{display:block;font-size:13px;color:var(--muted);margin-top:7px}nav{position:sticky;top:0;z-index:10;background:#f7f8f3f5;border-block:1px solid var(--line);backdrop-filter:blur(12px)}nav .wrap{display:flex;gap:26px;overflow:auto}nav a{white-space:nowrap;font-size:13px;text-decoration:none;padding:16px 0;color:var(--muted)}nav a:hover,nav a.active{color:var(--green);box-shadow:inset 0 -2px var(--green)}main section{padding:49px 0;border-bottom:1px solid var(--line)}.lede{max-width:850px;color:var(--muted);font-size:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:24px 0}.card,figure{background:#fff;border:1px solid var(--line);border-radius:10px;padding:25px;margin:24px 0}.grid .card,.grid figure{margin:0}.caption,figcaption{font-size:12px;color:var(--muted);line-height:1.55}figcaption{margin-top:16px}.caption{margin-bottom:23px}.decision{border-left:3px solid var(--green);padding:14px 20px;background:#eaf1eb;margin:24px 0}.caution{border-left-color:var(--orange);background:#f8eee6}.decision p:last-child{margin-bottom:0}.bar-row{display:grid;grid-template-columns:180px 1fr 60px;align-items:center;gap:14px;margin:15px 0;font-size:14px}.bar-row strong{text-align:right;font-variant-numeric:tabular-nums}.track{height:13px;background:#eef1ed;border-radius:3px;overflow:hidden}.bar{height:100%;background:var(--bar-color);border-radius:3px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;text-align:left}th,td{padding:12px 15px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:12px;color:var(--muted);font-weight:600;background:#edf1eb}td:not(:first-child){font-variant-numeric:tabular-nums}details{margin:20px 0;background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px 20px}summary{cursor:pointer;font-weight:600;font-size:14px}details[open] summary{margin-bottom:20px}.sources{font-size:12px;color:var(--muted);display:flex;gap:18px;flex-wrap:wrap;margin-top:20px}svg,img{width:100%;height:auto;display:block}svg text{font:12px 'Segoe UI',Arial,sans-serif;fill:#5a6964}.pill{display:inline-block;font-size:11px;background:#e4eee7;color:#235f4f;border-radius:20px;padding:3px 9px;margin-bottom:9px}.pill.no{background:#f6e9e0;color:#924020}.flow{display:flex;align-items:center;gap:12px;font-size:14px;font-weight:600;flex-wrap:wrap;margin:22px 0}.flow span{padding:11px 14px;border:1px solid var(--line);background:#fff;border-radius:6px}.flow i{font-style:normal;color:var(--green)}ul{padding-left:20px}li{margin:8px 0}code{font-size:.9em;overflow-wrap:anywhere}footer{padding:30px 0 55px;color:var(--muted);font-size:12px}.legend{font-size:13px;color:var(--muted);margin:20px 0}.end{background:#203e37;color:#fff;border-radius:10px;padding:28px;margin-top:24px}.end p{color:#d6e6dc}.end h3{font-size:22px}.end p:last-child{margin-bottom:0}
@media(max-width:700px){.wrap{padding:0 20px}header{padding-top:35px}h1{letter-spacing:-1.3px}.hero-stats,.grid{grid-template-columns:1fr}.hero-stats{gap:10px}.stat{padding:16px 20px}.stat strong{font-size:30px}nav .wrap{gap:22px}h2{font-size:28px}main section{padding:35px 0}.card,figure{padding:18px}.bar-row{grid-template-columns:1fr 54px;gap:5px 10px}.bar-row>span{grid-column:1/-1}.track{height:12px}th,td{padding:10px;min-width:90px}.intro{font-size:17px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{nav{position:static}.wrap{max-width:none}details{break-inside:avoid}figure,.card,.decision{break-inside:avoid}body{background:white}header{padding-top:20px}.hero-stats{grid-template-columns:repeat(3,1fr)}}
</style></head><body>
<header><div class="wrap"><div class="eyebrow">Yad2 · AI engineer assignment · Research notes</div>
<h1>From frozen features<br>to fine-tuned search.</h1>
<p class="intro">A training journey in turning short, messy Hebrew searches into structured filters. What improved, what failed, and what the evidence actually supports.</p>
<p class="meta">Recorded experiments · 16–17 November 2026 · About 7 minutes to read</p>
<div class="hero-stats"><div class="stat"><strong>@@CLEAN@@</strong><span>Complete JSON exact · 2,181 test queries</span></div><div class="stat"><strong>+9.7 pts</strong><span>Over Frozen Fusion · same test queries</span></div><div class="stat"><strong>@@COVERAGE@@</strong><span>Answered locally · @@ACCEPTED@@ accepted accuracy on test</span></div></div>
</div></header>
<nav aria-label="Journey chapters"><div class="wrap"><a href="#task">01 / The task</a><a href="#features">02 / Frozen features</a><a href="#extraction">03 / The plateau</a><a href="#finetuning">04 / Fine-tuning</a><a href="#routing">05 / Trust & routing</a><a href="#lessons">06 / Lessons</a></div></nav>
<main class="wrap">
<section id="task"><div class="kicker">01 / Define success</div><h2>The whole answer has to be right.</h2>
<p class="lede">A search parser must identify a domain, detect the requested fields, find their values, and normalize everything into <code>{category, params}</code>. Getting the city right while missing the price cap still makes the full answer wrong.</p>
<div class="flow" aria-label="Parsing pipeline"><span>Hebrew query</span><i aria-hidden="true">→</i><span>Domain + fields</span><i aria-hidden="true">→</i><span>Spans + values</span><i aria-hidden="true">→</i><span>Normalized JSON</span></div>
<p>For example, <bdi dir="rtl" lang="he">דירת 4 חדרים עד 2 מיליון</bdi> asks for four rooms and a maximum price. The bound belongs to the price field; it must not change the room count.</p>
<div class="grid"><div class="card"><h3>Three different measurements</h3><p><strong>Field-set exact:</strong> all requested fields detected, with no extras.</p><p><strong>JSON exact:</strong> domain, fields, and normalized values all correct.</p><p><strong>Routing:</strong> coverage is the share answered locally; accepted accuracy measures only those answers.</p></div><div class="card"><h3>Keep the data roles separate</h3><p><strong>Train</strong> fits models. <strong>Development</strong> chooses configurations. <strong>Calibration</strong> sets thresholds; an audit checks them.</p><p><strong>Test</strong> checks a frozen configuration. The tagger uses 17,272 train rows, 3,436 validation rows and the same 2,181 test rows used for the earlier comparison.</p></div></div>
<p class="legend">Reading rule: compare within a labeled chart or table. A field-detection score, a search-only development score, and an all-query test JSON score are different measurements.</p>
<div class="sources">@@DATA_SOURCE@@</div></section>

<section id="features"><div class="kicker">02 / Frozen Fusion · experiments 1–3</div><h2>Semantics needed surface detail.</h2>
<p class="lede">Could a frozen sentence encoder detect the complete filter set? E5 provided a baseline, but lexical identity—brands, amenities, morphology—was being blurred.</p>
@@REP_CHART@@
<p>Word and character TF-IDF features lifted test slot exact-set accuracy by <strong>14.2 points</strong> over slot-trained E5. Combining both representations added another <strong>3.4 points</strong>. Family-level bootstrap intervals excluded zero: +11.9 to +16.5 points for TF-IDF over E5, and +1.8 to +5.0 for fusion over TF-IDF.</p>
<div class="decision"><p><strong>Keep the complementary representations.</strong> TF-IDF preserves surface cues; E5 helps recover semantic paraphrases, long queries, and spelled-out numeric concepts. TF-IDF alone remained the lower-latency option, about six times faster per query in the recorded benchmark.</p></div>
<details><summary>Label correction and the full comparison</summary><p>Seven point/range twin pairs reduced 77 raw fields to 70 detection slots. Schema v2 then folded range names into point fields and canonicalized the vehicle-year label, leaving 69 fields. This removed a naming inconsistency; bounds remain in the values. The original-slot and v2 results below are separate evaluations.</p>@@REP_TABLE@@<p class="caption">Dev here means the original 1,019-row val_eval set. The v2 result keys retain the historical name “slot_exact_set”; they measure canonical field sets in that evaluation.</p></details>
<details><summary>Original figure: where confidence was useful</summary>@@RISK_IMAGE@@<p>Domain confidence was near chance at predicting whether the whole field set was right. Prediction margins ranked correctness better, motivating a dedicated correctness gate.</p></details>
<div class="sources">@@REP_SOURCE@@ @@V2_SOURCE@@</div></section>

<section id="extraction"><div class="kicker">03 / Frozen Fusion · experiments 4–7</div><h2>Fix the bottleneck, then recognize the plateau.</h2>
<p class="lede">Once fields were known, extraction was relatively strong. The larger loss came from fields the classifier never selected.</p>
<div class="grid"><div class="card"><h3>96.9% of supplied-field values correct</h3><p>The first extraction evaluation supplied gold fields for 1,019 development queries. Value accuracy was 96.9%; 92.2% of searches had every value correct.</p><p>The full pipeline achieved 81.1% JSON exact. Extraction could remove false-positive fields as well as introduce value errors, so JSON exact did not simply equal field accuracy minus extraction errors.</p></div><div class="card"><h3>Targeted fixes helped</h3><p>Field-local operators, Hebrew number reading, coherent ranges, negation, and safer fuzzy matching improved extraction. Cities such as <bdi dir="rtl" lang="he">באר שבע</bdi> stopped producing accidental numbers.</p><p>With the later fixes, supplied-field value accuracy reached 97.3% on development.</p></div></div>
<div class="decision caution"><p><strong>Rejected: random typo augmentation.</strong> One or two noisy copies per training row gave at most about one point on routing typo queries, offset by losses elsewhere. Real failures involved new phrasings and structured spelling errors.</p></div>
@@EXTRA_TABLE@@
<p class="caption">Experiment 7 · development only · 970 search rows in val_eval and 2,261 search rows in routing dev. The extractor is held constant within this table. These are not directly comparable to all-query test figures.</p>
<p>Extra training data improved coverage of new labels, but routing-dev JSON accuracy was flat from half to all of the extra batch. More of the same data was giving diminishing returns.</p>
<details><summary>Original figure: the first extraction error breakdown</summary>@@EXTRACT_IMAGE@@</details>
<div class="sources">@@EXTRACT_SOURCE@@ @@FIX_SOURCE@@ @@TYPO_SOURCE@@ @@EXTRA_SOURCE@@</div></section>

<section id="finetuning"><div class="kicker">04 / Fine-Tuned Tagger · experiment 1</div><h2>Learn the fields and their spans together.</h2>
<p class="lede">The next phase fine-tuned DictaBERT-base end to end for domain, field presence, BIO spans, operators, and implied values. Deterministic normalization stayed in the pipeline.</p>
@@JSON_CHART@@
<p>The final configuration used a <strong>three-seed ensemble</strong>, an encoder learning rate of <strong>5e-5</strong>, eight epochs, improved number normalization, and an extractor fallback for non-numeric fields with presence probability ≥ 0.5. Runtime query cleanup added a further eight net correct test answers: ten fixed and two broken.</p>
<figure><h3>More data helped; each increment helped less</h3><p class="caption">DictaBERT single models · val_eval · 1,617 queries · JSON exact before the final decoding improvements</p>@@CURVE@@<figcaption>Means over two family-grouped subset/seed runs at smaller sizes and three seeds at full size. The curve supports additional data, but does not establish gains on future test traffic.</figcaption></figure>
<div class="grid"><div class="card"><span class="pill">Adopted</span><h3>Better decoding, modest ensembling</h3><p>Normalizer fixes raised mean dev JSON from 89.3% to 90.2%. The fallback brought it to 90.7%. The final ensemble reached 91.2% dev and 84.8% test before runtime cleanup.</p></div><div class="card"><span class="pill no">Rejected</span><h3>More capacity was not the answer</h3><p>The categorical value head reduced accuracy. DictaBERT-large reached 89.5% at best, below the comparable 90.4% base model, at about 1.8× the epoch time.</p></div></div>
<details><summary>Learning-rate sweep, data sizes, and difficult test slices</summary>@@LR_TABLE@@<h3>Learning-curve data</h3>@@LEARNING_TABLE@@<h3>Remaining hard cases</h3>@@SLICE_TABLE@@<p class="caption">Slices describe the E1 final ensemble before cleanup. “Typos” here excludes three separately labeled typo+standard rows. The gold/silver gap reflects difficulty and distribution differences; it is not itself a label-error estimate.</p></details>
<div class="sources">@@BERT_SOURCE@@ @@SUMMARY_SOURCE@@ @@CLEAN_SOURCE@@</div></section>

<section id="routing"><div class="kicker">05 / Fine-Tuned Tagger · experiments 2–3</div><h2>Better answers still need a calibrated exit.</h2>
<p class="lede">Frozen Fusion became more valuable as a second opinion than as a replacement answer. Agreement between the two systems helped a learned gate decide which queries to answer locally.</p>
@@GATE_TABLE@@
<p class="caption">E1/E2 rows use the pre-registered 95% target. E2 is calibrated on validation; E3 uses a separate routing fit/selection/audit protocol. Lower bounds shown for the routing audit are computed over accepted units. These rows describe different evaluation sets, not one continuous improvement curve.</p>
<div class="decision caution"><p><strong>A target is not a guarantee.</strong> The E2 gate answers 84.5% of test queries locally, but its accepted accuracy is 92.9%, below the 95% target. The E3 routing-calibrated gate clears the 95% audit lower bound, at just 39.1% coverage. That certification applies to this audit protocol and population; the gold subset alone is too small to support the same claim.</p></div>
<div class="grid"><div class="card"><span class="pill no">Rejected</span><h3>Arbitrating disputed fields</h3><p>A per-field model tried choosing between tagger and fusion outputs. Its apparent dev gain failed on val_calib: 90.5% → 90.4%, with five rows fixed and seven broken.</p><p>Keep fusion as a gate signal; do not use that arbiter to rewrite answers.</p></div><div class="card"><span class="pill">Human check</span><h3>A blind audit of 120 labels</h3><p>The user reviewed 80 bulk-approved silver labels and 40 synthetic-v2 labels. One label was explicitly wrong; the validation-weighted error estimate was @@AUDIT_ERROR@@.</p><p>Including missing-field notes gives roughly 2–3%. This small audit does not establish an exact label-quality ceiling or prove labels are never a bottleneck.</p></div></div>
<details><summary>Routing caveats, historical costs, and data-role provenance</summary><p>The recorded E2 cost scenario assumes about 33 million monthly searches and roughly $0.10 per 1,000 LLM calls: approximately $505/month in fallback calls versus $3,250 for LLM-only. These are historical rounded assumptions and exclude local inference infrastructure.</p><p>A routed system’s total accuracy requires measuring the fallback on the rejected queries. Applying an LLM’s average accuracy to that harder subset gives an estimate, not an observed result.</p><p>Earlier Frozen Fusion experiment 5 used routing fit/selection rows as development data and explicitly required a new calibration batch afterward. The later tagger experiment 3 reused the recorded routing partition and reports a unit-based audit. Read its certification in that documented context; do not present the entire routing dataset as untouched across the whole research history.</p><p>The hybrid-fuel → vehicle-type product rule remains off by default: enabling it conflicts with the current labels. Its adoption requires changing the labeling convention too.</p></details>
<div class="sources">@@GATE_SOURCE@@ @@CALIB_SOURCE@@ @@AUDIT_SOURCE@@</div></section>

<section id="lessons"><div class="kicker">06 / Decisions to carry forward</div><h2>Accuracy improved. Trust remains conditional.</h2>
<div class="grid"><div class="card"><h3>Keep</h3><ul><li>Fine-tuned Hebrew tagging with deterministic normalization and a field-local fallback.</li><li>Complementary models as signals for correctness.</li><li>Family-aware splits, explicit metric definitions, and independent threshold audits.</li></ul></div><div class="card"><h3>Do next</h3><ul><li>Collect more realistic hard queries, especially typos and real estate; prioritize gold-like coverage.</li><li>Follow the documented fresh-data generation plan before claiming stronger routing guarantees.</li><li>Measure fallback outputs on actual rejected queries and total inference cost.</li></ul></div></div>
<div class="end"><h3>The main result</h3><p>Full JSON test accuracy rose from 75.4% with Frozen Fusion to @@CLEAN@@ with the cleaned tagger. A second model improved routing coverage, while independent checks exposed the gap between a useful operating point and a certified accuracy target.</p></div>
<details><summary>Evidence index and reproduction</summary><p>This page summarizes existing results; building it does not train a model or evaluate held-out queries.</p><p>Rebuild from the repository root: <code>python -m training.build_journey</code>.</p><ul><li>@@INDEX_FF@@</li><li>@@INDEX_BERT@@</li><li>@@INDEX_E2@@</li><li>@@INDEX_E3@@</li><li>@@INDEX_DATA@@</li></ul><p>Datasets, manifests, saved scores, and historical logs retain their recorded contents. Reproduction commands in the experiment READMEs use the new <code>training/</code> paths. Saved weights and some intermediate artifacts are ignored by Git and must be available locally to reproduce model inference.</p></details>
</section></main><footer><div class="wrap">Yad2 training journey · Built from local experiment records · No external fonts, chart services, or network requests.<br>Figures are embedded; evidence links resolve beside this page in the repository.</div></footer>
<script>
const links=[...document.querySelectorAll('nav a')];
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting){for(const link of links){const active=link.hash==='#'+entry.target.id;link.classList.toggle('active',active);if(active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');}}}},{rootMargin:'-15% 0px -60% 0px'});document.querySelectorAll('main section').forEach(s=>observer.observe(s));}
</script></body></html>'''
    replacements = {
        'CLEAN': pct(clean['test']['json_exact_clean']), 'COVERAGE': pct(chosen['test']['coverage']),
        'ACCEPTED': pct(chosen['test']['accuracy']), 'AUDIT_ERROR': pct(audit['weighted_error_rate']),
        'REP_CHART': representation_chart, 'REP_TABLE': representation_table, 'JSON_CHART': json_chart,
        'CURVE': curve(points), 'LR_TABLE': lr_table, 'LEARNING_TABLE': learning_table,
        'GATE_TABLE': gate_table, 'SLICE_TABLE': slice_table, 'EXTRA_TABLE': extra_table,
        'DATA_SOURCE': source(f'{BERT}/experiment1/README.md', 'Data and split roles'),
        'REP_SOURCE': source(f'{FF}/comparison/REPORT.html', 'Representation comparison'),
        'V2_SOURCE': source(f'{FF}/comparison/results/comparison_v2.json', 'Schema v2 results'),
        'RISK_IMAGE': archived_figure(f'{FF}/comparison/results/plots/risk_coverage_test.png', 'Original frozen-feature comparison: risk versus coverage on test; field-set correctness, not full JSON.'),
        'EXTRACT_IMAGE': archived_figure(f'{FF}/experiment_4_extraction/results/plots/extraction_gap.png', 'Original extraction evaluation: field-detection and value errors on development queries.'),
        'EXTRACT_SOURCE': source(f'{FF}/experiment_4_extraction/REPORT.html', 'Extraction evaluation'),
        'FIX_SOURCE': source(f'{FF}/experiment_5_routing_errors/README.md', 'Routing errors and fixes'),
        'TYPO_SOURCE': source(f'{FF}/experiment_6_typo_augment/README.md', 'Typo augmentation'),
        'EXTRA_SOURCE': source(f'{FF}/experiment_7_extra_train/README.md', 'Extra training data'),
        'BERT_SOURCE': source(f'{BERT}/experiment1/README.md', 'Fine-tuning experiments'),
        'SUMMARY_SOURCE': source(f'{BERT}/experiment1/results/summary.json', 'Fine-tuning metrics'),
        'CLEAN_SOURCE': source(f'{BERT}/experiment2/results/clean_report_with_test.json', 'Cleanup results'),
        'GATE_SOURCE': source(f'{BERT}/experiment2/README.md', 'Second-opinion gate'),
        'CALIB_SOURCE': source(f'{BERT}/experiment3/README.md', 'Routing calibration and audit'),
        'AUDIT_SOURCE': source(f'{BERT}/experiment3/audit/audit_result.json', 'Label audit results'),
        'INDEX_FF': source(f'{FF}/comparison/REPORT.html', 'Frozen representations: comparison and confidence intervals'),
        'INDEX_BERT': source(f'{BERT}/experiment1/README.md', 'DictaBERT: configuration, experiments, and final test'),
        'INDEX_E2': source(f'{BERT}/experiment2/README.md', 'Cleanup and Frozen Fusion as a second opinion'),
        'INDEX_E3': source(f'{BERT}/experiment3/README.md', 'Arbiter, calibration, and blind label audit'),
        'INDEX_DATA': source('../docs/data_generation_plan.md', 'Next-data generation plan'),
    }
    for key, value in replacements.items():
        html = html.replace(f'@@{key}@@', value)
    assert '@@' not in html, 'Unresolved template placeholder'
    (HERE / 'journey.html').write_text(html, encoding='utf-8', newline='\n')
    print(f'Built {HERE / "journey.html"}')


if __name__ == '__main__':
    build()
