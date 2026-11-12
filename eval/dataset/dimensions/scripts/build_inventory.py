#!/usr/bin/env python3
"""Build a field inventory from yad2_search_taxonomy.json.

Walks the taxonomy and emits one inventory entry per field, including the
hierarchy (path + the selector choices that lead to the field). Output:

    inventory/field_inventory.json
    inventory/field_inventory.html

The output is deterministic: entries follow the source document order, there
are no timestamps, and the source is identified by its SHA-256.

Node shapes recognised (by structure, not by key name):

    {"טיפוס": ..., "טווח": [a, b], ...}   -> typed field (optional range/extras)
    {"min": a, "max": b}                  -> range field
    ["x", "y", ...]                       -> enum field
    [a, b]  (exactly two numbers)         -> range field
    {"k": "v", ...}  (all values strings) -> mapping (e.g. typo map)
    {"A": {...}, "B": {...}}  (all children are containers)
                                          -> selector: choosing A/B is a field,
                                             then recurse with that condition
    {"A": [...], ...} whose keys are values of a sibling enum
                                          -> dependent enum, conditioned on sibling
    any other dict                        -> field group, recurse

Top-level scalars (version/date) go to "meta"; top-level dicts other than the
categories (e.g. "כללי", global cleanup config) are skipped.

Only two structural anchors are named explicitly: the categories key (its
children are the domains) and the field-spec keys (type/range/examples).

Usage:
    python scripts/build_inventory.py [--taxonomy PATH] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any

# --- Structural anchors in the taxonomy ------------------------------------
CATEGORIES_KEY = "קטגוריות"
SPEC_TYPE_KEY = "טיפוס"
SPEC_RANGE_KEY = "טווח"
SPEC_EXAMPLES_KEY = "דוגמאות"

# Hebrew type names -> canonical type names.
TYPE_MAP = {
    "מחרוזת": "string",
    "מספר": "number",
    "בוליאני": "boolean",
    "תאריך": "date",
    "תאריך_יחסי_או_מדויק": "date_relative_or_exact",
}

ROOT = Path(__file__).resolve().parent.parent


# --- Shape predicates -------------------------------------------------------
def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_field_spec(node: Any) -> bool:
    return isinstance(node, dict) and SPEC_TYPE_KEY in node


def is_range_dict(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and bool(node)
        and set(node) <= {"min", "max"}
        and all(_is_num(v) for v in node.values())
    )


def is_range_pair(node: Any) -> bool:
    return isinstance(node, list) and len(node) == 2 and all(_is_num(v) for v in node)


def is_mapping(node: Any) -> bool:
    return isinstance(node, dict) and bool(node) and all(isinstance(v, str) for v in node.values())


def is_container(node: Any) -> bool:
    """A dict that is not itself a leaf field."""
    return isinstance(node, dict) and not (is_field_spec(node) or is_range_dict(node) or is_mapping(node))


def is_selector(node: Any) -> bool:
    return isinstance(node, dict) and bool(node) and all(is_container(v) for v in node.values())


def dependent_parent(node: Any, siblings: dict) -> str | None:
    """If node is {key: [..]} and its keys are values of a sibling enum, return that sibling."""
    if not (isinstance(node, dict) and node and all(isinstance(v, list) for v in node.values())):
        return None
    for name, sib in siblings.items():
        if isinstance(sib, list) and all(isinstance(x, str) for x in sib) and set(node) <= set(sib):
            return name
    return None


# --- Inventory builder ------------------------------------------------------
class InventoryBuilder:
    def __init__(self) -> None:
        self.entries: list[dict] = []
        self.warnings: list[str] = []

    def add(self, *, domain: str, path: list[str], kind: str, type_: str, type_raw: str | None = None,
            values: list | None = None, range_: dict | None = None, mapping: dict | None = None,
            conditions: dict | None = None, group: str | None = None, extra: dict | None = None,
            field_name: str | None = None) -> None:
        self.entries.append({
            "id": "/".join(path),
            "domain": domain,
            "field_name": field_name or path[-1],
            "path": list(path),
            "hierarchy": format_hierarchy(path, conditions or {}),
            "group": group,
            "kind": kind,
            "type": type_,
            "type_raw": type_raw,
            "possible_values": values,
            "range": range_,
            "mapping": mapping,
            "conditions": dict(conditions or {}),
            "extra": extra or {},
        })

    # A single leaf/branch node inside a domain.
    def visit(self, domain: str, key: str, node: Any, path: list[str], conditions: dict,
              group: str | None, siblings: dict, kind: str = "field") -> None:
        p = path + [key]

        if is_field_spec(node):
            raw = node[SPEC_TYPE_KEY]
            extra = {k: v for k, v in node.items() if k not in (SPEC_TYPE_KEY, SPEC_RANGE_KEY, SPEC_EXAMPLES_KEY)}
            if SPEC_EXAMPLES_KEY in node:
                extra = {"examples": node[SPEC_EXAMPLES_KEY], **extra}
            rng = None
            if SPEC_RANGE_KEY in node:
                r = node[SPEC_RANGE_KEY]
                if is_range_pair(r):
                    rng = {"min": r[0], "max": r[1]}
                else:
                    self.warnings.append(f"{'/'.join(p)}: malformed range {r!r}")
            canon = TYPE_MAP.get(raw)
            if canon is None:
                self.warnings.append(f"{'/'.join(p)}: unknown type {raw!r}")
                canon = f"unknown:{raw}"
            self.add(domain=domain, path=p, kind=kind, type_=canon, type_raw=raw, range_=rng,
                     conditions=conditions, group=group, extra=extra)

        elif is_range_dict(node):
            self.add(domain=domain, path=p, kind=kind, type_="range",
                     range_={"min": node.get("min"), "max": node.get("max")},
                     conditions=conditions, group=group)

        elif is_range_pair(node):
            self.add(domain=domain, path=p, kind=kind, type_="range",
                     range_={"min": node[0], "max": node[1]}, conditions=conditions, group=group)

        elif isinstance(node, list):
            if all(isinstance(x, str) for x in node):
                type_ = "enum"
            elif all(_is_num(x) for x in node):
                type_ = "enum_number"
            else:
                type_ = "enum_mixed"
                self.warnings.append(f"{'/'.join(p)}: list with mixed item types")
            self.add(domain=domain, path=p, kind=kind, type_=type_, values=list(node),
                     conditions=conditions, group=group)

        elif (parent := dependent_parent(node, siblings)) is not None:
            for parent_value, values in node.items():
                self.add(domain=domain, path=p + [parent_value], field_name=key, kind="dependent_field",
                         type_="enum", values=list(values), conditions={**conditions, parent: parent_value},
                         group=group, extra={"depends_on": parent})

        elif is_mapping(node):
            self.add(domain=domain, path=p, kind="mapping", type_="mapping", mapping=dict(node),
                     conditions=conditions, group=group)

        elif is_selector(node):
            self.add(domain=domain, path=p, kind="selector", type_="enum", values=list(node),
                     conditions=conditions, group=group)
            for child_key, child in node.items():
                self.visit_container(domain, child, p + [child_key], {**conditions, key: child_key}, None)

        elif isinstance(node, dict):
            self.visit_container(domain, node, p, conditions, key)

        elif isinstance(node, (str, int, float, bool)) or node is None:
            scalar_type = ("null" if node is None else "boolean" if isinstance(node, bool)
                           else "number" if _is_num(node) else "string")
            self.add(domain=domain, path=p, kind="constant",
                     type_=scalar_type, values=[node],
                     conditions=conditions, group=group)

        else:  # pragma: no cover - JSON can't produce anything else
            self.warnings.append(f"{'/'.join(p)}: unrecognised node {type(node).__name__}")

    def visit_container(self, domain: str, node: dict, path: list[str], conditions: dict,
                        group: str | None, kind: str = "field") -> None:
        for key, child in node.items():
            self.visit(domain, key, child, path, conditions, group, node, kind)

    def build(self, taxonomy: dict) -> dict:
        meta: dict[str, Any] = {}
        for key, node in taxonomy.items():
            if key == CATEGORIES_KEY:
                if not is_selector(node):
                    raise ValueError(f"'{CATEGORIES_KEY}' must be a dict of domain objects")
                for domain, body in node.items():
                    self.visit_container(domain, body, [domain], {}, None)
            elif not isinstance(node, dict):
                meta[key] = node
            # other top-level dicts (e.g. "כללי": global cleanup config) are not search fields
        return meta


def format_hierarchy(path: list[str], conditions: dict) -> str:
    """Render e.g. 'רכב › יצרנים[טויוטה] › תתי_דגמים[קורולה]'."""
    chosen = set(conditions.values())
    parts: list[str] = []
    for seg in path:
        if seg in chosen and parts:
            parts[-1] = f"{parts[-1]}[{seg}]"
        else:
            parts.append(seg)
    return " › ".join(parts)


def summarize(entries: list[dict]) -> dict:
    domains: dict[str, dict] = {}
    for e in entries:
        d = domains.setdefault(e["domain"], {"entries": 0, "by_kind": {}, "by_type": {}, "field_names": []})
        d["entries"] += 1
        d["by_kind"][e["kind"]] = d["by_kind"].get(e["kind"], 0) + 1
        d["by_type"][e["type"]] = d["by_type"].get(e["type"], 0) + 1
        if e["kind"] in ("field", "selector", "dependent_field") and e["field_name"] not in d["field_names"]:
            d["field_names"].append(e["field_name"])
    return {"total_entries": len(entries), "domains": domains}


# --- HTML rendering ---------------------------------------------------------
def esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def render_values(e: dict) -> str:
    if e["mapping"]:
        return "".join(f'<span class="chip map">{esc(k)} → {esc(v)}</span>' for k, v in e["mapping"].items())
    if e["possible_values"]:
        return "".join(f'<span class="chip">{esc(v)}</span>' for v in e["possible_values"])
    return '<span class="na">—</span>'


def render_range(e: dict) -> str:
    r = e["range"]
    if not r:
        return '<span class="na">—</span>'
    lo = "" if r.get("min") is None else f"{r['min']:,}"
    hi = "" if r.get("max") is None else f"{r['max']:,}"
    return f'<span class="num" dir="ltr">{esc(lo)} – {esc(hi)}</span>'


def render_details(e: dict) -> str:
    bits = []
    if e["conditions"]:
        bits.append("תנאי: " + ", ".join(f"{esc(k)}={esc(v)}" for k, v in e["conditions"].items()))
    if e["group"]:
        bits.append(f"קבוצה: {esc(e['group'])}")
    for k, v in e["extra"].items():
        val = ", ".join(map(str, v)) if isinstance(v, list) else v
        bits.append(f"{esc(k)}: {esc(val)}")
    return "<br>".join(bits) or '<span class="na">—</span>'


def render_html(inv: dict) -> str:
    entries = inv["entries"]
    domains = list(inv["summary"]["domains"])
    rows = []
    for e in entries:
        depth = len(e["path"]) - 1
        rows.append(
            f'<tr data-domain="{esc(e["domain"])}" data-kind="{esc(e["kind"])}">'
            f'<td><span class="dom">{esc(e["domain"])}</span></td>'
            f'<td class="fname" style="padding-inline-start:{8 + depth * 10}px">{esc(e["field_name"])}</td>'
            f'<td class="hier">{esc(e["hierarchy"])}</td>'
            f'<td><span class="kind k-{esc(e["kind"])}">{esc(e["kind"])}</span></td>'
            f'<td><code dir="ltr">{esc(e["type"])}</code>'
            + (f'<div class="raw">{esc(e["type_raw"])}</div>' if e["type_raw"] else "")
            + f'</td><td class="vals">{render_values(e)}</td>'
            f'<td>{render_range(e)}</td><td class="det">{render_details(e)}</td></tr>'
        )
    chips = "".join(
        f'<button class="filter" data-d="{esc(d)}">{esc(d)} '
        f'<span>{inv["summary"]["domains"][d]["entries"]}</span></button>' for d in domains
    )
    kinds = sorted({e["kind"] for e in entries})
    kind_opts = "".join(f'<option value="{esc(k)}">{esc(k)}</option>' for k in kinds)
    src = inv["source"]
    return f"""<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Yad2 Field Inventory</title>
<style>
:root {{
  --bg:#f7f7f5; --panel:#fff; --fg:#1d1d1b; --muted:#6b6b66; --line:#e3e2dd;
  --chip:#eef0f3; --chip-fg:#2a3340; --accent:#2f5bd3; --map:#fdf1e3;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141413; --panel:#1d1d1b; --fg:#ecebe6; --muted:#9a9993; --line:#33322f;
          --chip:#2a2d33; --chip-fg:#d6dae2; --accent:#7ea0ff; --map:#3a2e20; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",Arial,sans-serif; }}
header {{ padding:20px 16px 8px; max-width:1400px; margin:auto; }}
h1 {{ margin:0 0 4px; font-size:22px; }}
.meta {{ color:var(--muted); font-size:12px; }}
.bar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:8px 16px;
  max-width:1400px; margin:auto; position:sticky; top:0; background:var(--bg); z-index:2; }}
.filter {{ border:1px solid var(--line); background:var(--panel); color:var(--fg);
  border-radius:999px; padding:4px 12px; cursor:pointer; font:inherit; }}
.filter span {{ color:var(--muted); font-size:12px; }}
.filter.on {{ border-color:var(--accent); color:var(--accent); }}
input, select {{ font:inherit; padding:5px 10px; border:1px solid var(--line); border-radius:8px;
  background:var(--panel); color:var(--fg); }}
input {{ flex:1; min-width:180px; }}
#count {{ color:var(--muted); font-size:12px; }}
.wrap {{ max-width:1400px; margin:auto; padding:0 16px 32px; overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line);
  border-radius:10px; overflow:hidden; }}
th, td {{ text-align:start; vertical-align:top; padding:8px; border-bottom:1px solid var(--line); }}
th {{ font-size:12px; color:var(--muted); font-weight:600; background:var(--panel); }}
.fname {{ font-weight:600; white-space:nowrap; }}
.hier {{ color:var(--muted); font-size:12px; min-width:180px; }}
.dom {{ white-space:nowrap; }}
.kind {{ font-size:11px; padding:1px 7px; border-radius:6px; background:var(--chip); white-space:nowrap; }}
.k-selector {{ outline:1px solid var(--accent); }}
.raw {{ color:var(--muted); font-size:11px; }}
.vals {{ max-width:420px; }}
.chip {{ display:inline-block; margin:2px; padding:1px 8px; border-radius:6px; background:var(--chip);
  color:var(--chip-fg); font-size:12px; }}
.chip.map {{ background:var(--map); }}
.na {{ color:var(--muted); }}
.num {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
.det {{ font-size:12px; color:var(--muted); min-width:140px; }}
code {{ font-size:12px; }}
</style>
</head>
<body>
<header>
  <h1>Yad2 Field Inventory</h1>
  <div class="meta">גרסת טקסונומיה {esc(inv["meta"].get("גרסה", "?"))} · עדכון {esc(inv["meta"].get("עדכון", "?"))}
  · {inv["summary"]["total_entries"]} רשומות · <span dir="ltr">sha256 {esc(src["sha256"][:12])}</span></div>
</header>
<div class="bar">
  <button class="filter on" data-d="">הכל <span>{len(entries)}</span></button>
  {chips}
  <select id="kind"><option value="">כל הסוגים</option>{kind_opts}</select>
  <input id="q" type="search" placeholder="חיפוש שדה, ערך או היררכיה…">
  <span id="count"></span>
</div>
<div class="wrap">
<table>
<thead><tr><th>Domain</th><th>Field Name</th><th>Hierarchy</th><th>Kind</th><th>Type</th>
<th>Possible Values</th><th>Range</th><th>Details</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
</div>
<script>
(function () {{
  var dom = "", rows = Array.prototype.slice.call(document.querySelectorAll("tbody tr"));
  var q = document.getElementById("q"), kind = document.getElementById("kind");
  var count = document.getElementById("count");
  function apply() {{
    var t = q.value.trim().toLowerCase(), k = kind.value, n = 0;
    rows.forEach(function (r) {{
      var ok = (!dom || r.dataset.domain === dom) && (!k || r.dataset.kind === k) &&
               (!t || r.textContent.toLowerCase().indexOf(t) !== -1);
      r.style.display = ok ? "" : "none"; if (ok) n++;
    }});
    count.textContent = n + " / " + rows.length;
  }}
  document.querySelectorAll(".filter").forEach(function (b) {{
    b.addEventListener("click", function () {{
      document.querySelectorAll(".filter").forEach(function (x) {{ x.classList.remove("on"); }});
      b.classList.add("on"); dom = b.dataset.d; apply();
    }});
  }});
  q.addEventListener("input", apply); kind.addEventListener("change", apply); apply();
}})();
</script>
</body>
</html>
"""


# --- Entry point ------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--taxonomy", type=Path, default=ROOT / "yad2_search_taxonomy.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "inventory")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any warnings are raised")
    args = ap.parse_args(argv)

    raw = args.taxonomy.read_bytes()
    taxonomy = json.loads(raw.decode("utf-8"))

    builder = InventoryBuilder()
    meta = builder.build(taxonomy)
    inventory = {
        "source": {"file": args.taxonomy.name, "sha256": hashlib.sha256(raw).hexdigest()},
        "meta": meta,
        "summary": summarize(builder.entries),
        "warnings": builder.warnings,
        "entries": builder.entries,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "field_inventory.json"
    html_path = args.out_dir / "field_inventory.html"
    json_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    html_path.write_text(render_html(inventory), encoding="utf-8", newline="\n")

    print(f"{len(builder.entries)} entries -> {json_path} , {html_path}")
    for w in builder.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 1 if (args.strict and builder.warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
