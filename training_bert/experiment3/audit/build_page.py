"""Build audit.html: a local, self-contained review page for sample.jsonl (no server, nothing published).

    python -m training_bert.experiment3.audit.build_page
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

PAGE = """<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label Audit</title>
<style>
:root { --bg:#f7f7f5; --card:#ffffff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e3e2dd; --accent:#2f5bd3;
        --ok:#1f7a46; --bad:#b3261e; --unsure:#9a6700; --chip:#efeee9; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#161615; --card:#1f1f1d; --ink:#ecebe6; --muted:#a3a29c; --line:#34332f; --accent:#7f9cf5;
  --ok:#5cc489; --bad:#f2786d; --unsure:#e2b13c; --chip:#2a2a27; } }
:root[data-theme="dark"] { --bg:#161615; --card:#1f1f1d; --ink:#ecebe6; --muted:#a3a29c; --line:#34332f;
  --accent:#7f9cf5; --ok:#5cc489; --bad:#f2786d; --unsure:#e2b13c; --chip:#2a2a27; }
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui, "Segoe UI", Arial, sans-serif; }
main { max-width:860px; margin:0 auto; padding:20px 16px 60px; }
header { display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between; margin-bottom:14px; }
h1 { font-size:18px; margin:0; }
.progress { color:var(--muted); font-size:14px; direction:ltr; }
.bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin:6px 0 18px; }
.bar > div { height:100%; background:var(--accent); width:0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:20px; }
.meta { display:flex; gap:8px; flex-wrap:wrap; font-size:13px; color:var(--muted); margin-bottom:10px; direction:ltr; justify-content:flex-end; }
.chip { background:var(--chip); border-radius:999px; padding:2px 10px; }
.query { font-size:22px; line-height:1.6; margin:4px 0 18px; word-break:break-word; }
table { width:100%; border-collapse:collapse; font-size:15px; }
td { border-top:1px solid var(--line); padding:7px 6px; vertical-align:top; }
td.k { color:var(--muted); width:38%; }
td.v { font-weight:600; word-break:break-word; }
.cat { font-size:15px; margin-bottom:8px; }
.none { color:var(--muted); font-style:italic; }
.actions { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
button { font:inherit; border:1px solid var(--line); background:var(--card); color:var(--ink); border-radius:8px;
         padding:9px 14px; cursor:pointer; }
button:hover { border-color:var(--accent); }
button.ok.sel { background:var(--ok); color:#fff; border-color:var(--ok); }
button.bad.sel { background:var(--bad); color:#fff; border-color:var(--bad); }
button.unsure.sel { background:var(--unsure); color:#fff; border-color:var(--unsure); }
textarea { width:100%; margin-top:12px; min-height:60px; font:inherit; padding:8px; border-radius:8px;
           border:1px solid var(--line); background:var(--bg); color:var(--ink); }
.nav { display:flex; justify-content:space-between; margin-top:14px; gap:10px; flex-wrap:wrap; }
.help { color:var(--muted); font-size:13px; margin-top:16px; direction:ltr; text-align:left; }
kbd { background:var(--chip); border-radius:4px; padding:0 5px; font-size:12px; }
.summary { margin-top:16px; font-size:14px; direction:ltr; text-align:left; color:var(--muted); }
</style>
</head>
<body>
<main>
  <header>
    <h1>Label audit — is this label correct?</h1>
    <div class="progress" id="progress"></div>
  </header>
  <div class="bar"><div id="bar"></div></div>
  <section class="card">
    <div class="meta" id="meta"></div>
    <div class="query" id="query"></div>
    <div class="cat" id="cat"></div>
    <table id="params"></table>
    <div class="actions">
      <button class="ok" id="b1">✓ Correct <kbd>1</kbd></button>
      <button class="bad" id="b2">✗ Wrong <kbd>2</kbd></button>
      <button class="unsure" id="b3">? Unsure / ambiguous <kbd>3</kbd></button>
    </div>
    <textarea id="note" placeholder="What is wrong / what the label should be (optional) — N to focus"></textarea>
    <div class="nav">
      <button id="prev">→ Previous</button>
      <button id="next-open">Next unreviewed</button>
      <button id="next">Next ←</button>
    </div>
  </section>
  <div class="nav">
    <button id="export">Export audit_verdicts.json</button>
  </div>
  <div class="summary" id="summary"></div>
  <div class="help">Judge the label against the query as a Yad2 search: every field should be there, nothing extra,
    values and bounds right, a non-search should have no category. Keys: <kbd>1</kbd> correct, <kbd>2</kbd> wrong,
    <kbd>3</kbd> unsure, <kbd>N</kbd> note, <kbd>←</kbd>/<kbd>→</kbd> move. Progress is kept in this browser;
    Export when done.</div>
</main>
<script>
const ROWS = __DATA__;
const KEY = "label-audit-v1";
let st = {}; try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { st = {}; }
let i = 0;
const $ = id => document.getElementById(id);
function save() { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
function fmt(v) {
  if (v === true) return "true"; if (v === false) return "false";
  if (Array.isArray(v)) return v.map(fmt).join(" | ");
  if (v && typeof v === "object") return Object.entries(v).map(([k, x]) => k + ": " + x).join(", ");
  return String(v);
}
function render() {
  const r = ROWS[i], s = st[r.id] || {};
  $("meta").innerHTML = `<span class="chip">#${r.n} / ${ROWS.length}</span><span class="chip">${r.pool}</span>` +
    `<span class="chip">${r.language}</span><span class="chip">${r.id}</span>`;
  $("query").textContent = r.query;
  $("cat").innerHTML = r.category ? `קטגוריה: <b>${r.category}</b>` : `<span class="none">לא חיפוש (no category)</span>`;
  const ps = Object.entries(r.params || {});
  $("params").innerHTML = ps.length ? ps.map(([k, v]) =>
    `<tr><td class="k">${k}</td><td class="v">${fmt(v).replace(/</g, "&lt;")}</td></tr>`).join("")
    : `<tr><td class="none">no fields</td></tr>`;
  ["b1", "b2", "b3"].forEach((b, j) => $(b).classList.toggle("sel", s.verdict === ["correct", "wrong", "unsure"][j]));
  $("note").value = s.note || "";
  const done = ROWS.filter(x => st[x.id] && st[x.id].verdict).length;
  $("progress").textContent = `${done} / ${ROWS.length} reviewed`;
  $("bar").style.width = (100 * done / ROWS.length) + "%";
  const c = {correct: 0, wrong: 0, unsure: 0};
  Object.values(st).forEach(x => x.verdict && c[x.verdict]++);
  $("summary").textContent = `correct ${c.correct} · wrong ${c.wrong} · unsure ${c.unsure}`;
}
function setV(v) {
  const r = ROWS[i];
  st[r.id] = {...(st[r.id] || {}), verdict: v, note: $("note").value, at: new Date().toISOString()};
  save(); if (i < ROWS.length - 1) i++; render();
}
function go(d) { i = Math.max(0, Math.min(ROWS.length - 1, i + d)); render(); }
$("b1").onclick = () => setV("correct"); $("b2").onclick = () => setV("wrong"); $("b3").onclick = () => setV("unsure");
$("prev").onclick = () => go(-1); $("next").onclick = () => go(1);
$("next-open").onclick = () => { const j = ROWS.findIndex(x => !(st[x.id] && st[x.id].verdict)); if (j >= 0) { i = j; render(); } };
$("note").oninput = () => { const r = ROWS[i]; st[r.id] = {...(st[r.id] || {}), note: $("note").value}; save(); };
$("export").onclick = () => {
  const out = ROWS.map(r => ({id: r.id, n: r.n, pool: r.pool, verdict: (st[r.id] || {}).verdict || null,
                              note: (st[r.id] || {}).note || "", at: (st[r.id] || {}).at || null}));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "audit_verdicts.json"; a.click();
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") { if (e.key === "Escape") e.target.blur(); return; }
  if (e.key === "1") setV("correct"); else if (e.key === "2") setV("wrong"); else if (e.key === "3") setV("unsure");
  else if (e.key === "ArrowLeft") go(1); else if (e.key === "ArrowRight") go(-1);
  else if (e.key.toLowerCase() === "n") { e.preventDefault(); $("note").focus(); }
});
render();
</script>
</body>
</html>
"""


def main():
    rows = [json.loads(l) for l in (HERE / "sample.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    (HERE / "audit.html").write_text(PAGE.replace("__DATA__", json.dumps(rows, ensure_ascii=False)), encoding="utf-8")
    print(f"wrote {HERE / 'audit.html'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
