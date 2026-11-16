"""Render REPORT.md as a self-contained REPORT.html (charts embedded where the report references them).

    python training/comparison/report_html.py

Handles the Markdown subset the report uses: headings, paragraphs, bullet/numbered lists (one nesting level),
pipe tables, **bold**, *italic*, `code`. Rerun after editing REPORT.md.
"""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "REPORT.md", HERE / "REPORT.html"
PLOT_RE = re.compile(r"`results/plots/([\w.-]+\.png)`")
CAPTIONS = {
    "headline.png": "Headline metrics: hollow = val_eval (development), filled = test (frozen configs).",
    "bootstrap_ci.png": "Paired differences in slot exact-set with family-level bootstrap 95% CIs.",
    "slot_delta.png": "Per-slot F1 change, Exp3 vs Exp1-slots (val_eval, development).",
    "slices_test.png": "Slot exact-set by slice on test.",
    "learning_curves.png": "Learning curves at each experiment's selected config.",
    "risk_coverage_test.png": "Risk–coverage on test: which score predicts a fully correct slot set.",
    "latency.png": "End-to-end latency per query by stage (CPU, batch 1).",
}


def inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<em>\1</em>", t)
    return t


def figure(name: str) -> str:
    data = base64.b64encode((HERE / "results" / "plots" / name).read_bytes()).decode()
    cap = CAPTIONS.get(name, name)
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(cap)}" loading="lazy">'
            f"<figcaption>{html.escape(cap)}</figcaption></figure>")


def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]
    numeric = re.compile(r"^[\s*~≈+−\-\[\]0-9.,/%×msMB²¹]+$")
    out = ['<div class="table-wrap"><table><thead><tr>']
    out += [f"<th>{inline(c)}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(
            f'<td class="{"num" if numeric.match(c) and i else ""}">{inline(c)}</td>' for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def convert(md: str) -> tuple[str, str, list[tuple[str, str]]]:
    lines = md.splitlines()
    body, toc, pending = [], [], []
    title = "Report"
    i = 0

    def flush_figs():
        while pending:
            body.append(figure(pending.pop(0)))

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            title = line[2:].strip()
            i += 1
            continue
        m = re.match(r"^(#{2,3}) (.*)", line)
        if m:
            flush_figs()
            level, text = len(m.group(1)), m.group(2)
            pending += PLOT_RE.findall(text)
            clean = PLOT_RE.sub("", text)
            clean = re.sub(r"\(\s*[,;]\s*", "(", clean)   # '(, val_eval' -> '(val_eval'
            clean = re.sub(r"\s*[,;]\s*\)", ")", clean)   # 'frozen; )' -> 'frozen)'
            clean = re.sub(r"\s*\(\s*\)", "", clean).strip(" ,;")
            anchor = re.sub(r"[^\w]+", "-", clean.lower()).strip("-")
            if level == 2:
                toc.append((anchor, clean))
            body.append(f'<h{level} id="{anchor}">{inline(clean)}</h{level}>')
            i += 1
            continue
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            body.append(table(rows))
            continue
        if re.match(r"^\s*([-*]|\d+\.) ", line):
            ordered = bool(re.match(r"^\d+\.", line.strip()))
            items, stack = [], []
            while i < len(lines) and lines[i].strip() and (re.match(r"^\s*([-*]|\d+\.) ", lines[i])
                                                           or lines[i].startswith("  ")):
                l = lines[i]
                mm = re.match(r"^(\s*)([-*]|\d+\.) (.*)", l)
                if mm:
                    items.append([len(mm.group(1)), mm.group(3)])
                else:
                    items[-1][1] += " " + l.strip()
                i += 1
            tag = "ol" if ordered else "ul"
            out, open_sub = [f"<{tag}>"], False
            for depth, text in items:
                pending += PLOT_RE.findall(text)
                text = inline(text)
                if depth >= 2 and not open_sub:
                    out[-1] = out[-1].removesuffix("</li>")
                    out.append("<ul>")
                    open_sub = True
                elif depth < 2 and open_sub:
                    out.append("</ul></li>")
                    open_sub = False
                out.append(f"<li>{text}</li>")
            if open_sub:
                out.append("</ul></li>")
            out.append(f"</{tag}>")
            body.append("".join(out))
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||\s*([-*]|\d+\.) )", lines[i]):
            para.append(lines[i].strip())
            i += 1
        text = " ".join(para)
        pending += PLOT_RE.findall(text)
        body.append(f"<p>{inline(text)}</p>")
    flush_figs()
    return title, "\n".join(body), toc


CSS = """
:root{--bg:#fcfcfb;--surface:#ffffff;--ink:#0b0b0b;--ink-2:#52514e;--muted:#8a8984;--line:#e4e3df;
--accent:#2a78d6;--code-bg:#f2f1ed;--row:#f7f6f3;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--surface:#222220;--ink:#f0efec;
--ink-2:#c3c2b7;--muted:#8f8e86;--line:#3a3936;--accent:#3987e5;--code-bg:#2c2b29;--row:#262624;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#1a1a19;--surface:#222220;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#8f8e86;--line:#3a3936;
--accent:#3987e5;--code-bg:#2c2b29;--row:#262624;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 "Segoe UI",system-ui,-apple-system,Arial,sans-serif}
.layout{display:grid;grid-template-columns:240px minmax(0,1fr);gap:40px;max-width:1240px;margin:0 auto;padding:32px 24px}
nav{position:sticky;top:24px;align-self:start;font-size:14px}
nav .label{color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-size:12px;margin-bottom:8px}
nav a{display:block;color:var(--ink-2);text-decoration:none;padding:4px 0 4px 10px;border-left:2px solid var(--line)}
nav a:hover{color:var(--accent);border-left-color:var(--accent)}
main{min-width:0}
h1{font-size:30px;line-height:1.25;margin:0 0 8px}
.sub{color:var(--ink-2);margin:0 0 28px}
h2{font-size:22px;margin:44px 0 12px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:18px;margin:28px 0 8px}
p,li{color:var(--ink)}
strong{font-weight:650}
code{font-family:Consolas,"Cascadia Mono",monospace;font-size:.88em;background:var(--code-bg);padding:1px 5px;
border-radius:4px;unicode-bidi:plaintext}
ul,ol{padding-left:22px}
li{margin:4px 0}
.table-wrap{overflow-x:auto;margin:14px 0 20px;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink-2);font-weight:600;background:var(--row);white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
figure{margin:18px 0 26px;background:#fcfcfb;border:1px solid var(--line);border-radius:8px;padding:12px}
figure img{display:block;max-width:100%;height:auto;margin:0 auto}
figcaption{color:#52514e;font-size:13px;margin-top:8px}
.theme{position:fixed;top:14px;right:16px;background:var(--surface);color:var(--ink-2);border:1px solid var(--line);
border-radius:6px;padding:4px 10px;font:13px inherit;cursor:pointer}
@media (max-width:900px){.layout{grid-template-columns:1fr;padding:20px 16px}nav{position:static}}
"""


def main():
    title, body, toc = convert(SRC.read_text(encoding="utf-8"))
    nav = "".join(f'<a href="#{a}">{inline(t)}</a>' for a, t in toc)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Experiment Comparison Report</title><style>{CSS}</style></head>
<body><button class="theme" onclick="var r=document.documentElement;r.dataset.theme=
(r.dataset.theme==='dark'||(!r.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches))?'light':'dark'">
Theme</button>
<div class="layout"><nav><div class="label">Contents</div>{nav}</nav>
<main><h1>{inline(title)}</h1>
<p class="sub">Generated from <code>training/comparison/REPORT.md</code> by <code>report_html.py</code>.</p>
{body}</main></div></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
