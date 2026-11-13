"""Streamlit editor for the ground-truth eval dataset.

Run:  streamlit run eval/dataset/app/app.py

Data:   eval/dataset/data/ground_truth.jsonl
Fields: eval/dataset/inventory/field_inventory.json   (generated from the taxonomy)
Dims:   eval/dataset/dimensions/dimensions_matrix.json
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:  # `streamlit run` adds it; other runners (AppTest) don't
    sys.path.insert(0, str(APP_DIR))

import store  # noqa: E402
from inventory_index import InventoryIndex, is_number, load_index  # noqa: E402

DATASET_DIR = APP_DIR.parent
INVENTORY_PATH = DATASET_DIR / "inventory" / "field_inventory.json"
DIMENSIONS_PATH = DATASET_DIR / "dimensions" / "dimensions_matrix.json"
DATA_PATH = DATASET_DIR / "data" / "ground_truth.jsonl"

NEW = "__new__"
NO_CATEGORY = "— ללא קטגוריה (null) —"
NUM_OPS = ["exact", "min", "max", "range"]

st.set_page_config(page_title="Ground Truth Editor", page_icon="📝", layout="wide")
st.markdown(
    """<style>
    .stMainBlockContainer, [data-testid="stSidebarContent"] { direction: rtl; text-align: right; }
    textarea, input { direction: rtl; }
    [data-testid="stJson"], code { direction: ltr; text-align: left; }
    </style>""",
    unsafe_allow_html=True,
)


# --- data -------------------------------------------------------------------
@st.cache_resource
def get_index(_inv_mtime: float, _dim_mtime: float) -> InventoryIndex:
    return load_index(INVENTORY_PATH, DIMENSIONS_PATH)


def blank() -> dict:
    return {"id": None, "query": "", "expected": {"category": None, "params": {}}, "metadata": {}, "notes": ""}


def load_records() -> list[dict]:
    return store.load(DATA_PATH)


index = get_index(INVENTORY_PATH.stat().st_mtime, DIMENSIONS_PATH.stat().st_mtime)
try:
    records = load_records()
except ValueError as exc:
    st.error(f"Cannot read {DATA_PATH}: {exc}")
    st.stop()
by_id = {r["id"]: r for r in records}

# --- session state ------------------------------------------------------------
ss = st.session_state
ss.setdefault("gen", 0)            # bumping it re-creates every editor widget from `loaded`
ss.setdefault("loaded", blank())   # record the editor was opened with (widget defaults)
ss.setdefault("current_id", None)  # None = new record
ss.setdefault("sel", NEW)
if "_pending_sel" in ss:           # set by actions that run after the sidebar was drawn
    ss.sel = ss.pop("_pending_sel")


def open_record(rid: str | None, record: dict | None = None) -> None:
    ss.current_id = rid
    ss.loaded = copy.deepcopy(record if record is not None else (by_id.get(rid) if rid else blank()))
    ss.gen += 1


def on_select() -> None:
    rid = None if ss.sel == NEW else ss.sel
    fresh = {r["id"]: r for r in load_records()}
    open_record(rid, fresh.get(rid) if rid else None)


def on_new() -> None:
    ss.sel = NEW
    open_record(None)


def finish(sel: str, flash: str) -> None:
    ss._pending_sel = sel
    ss._flash = flash
    st.rerun()


# --- sidebar: records + coverage ---------------------------------------------
with st.sidebar:
    st.header("📚 Ground truth")
    st.caption(f"{len(records)} רשומות · `{DATA_PATH.relative_to(DATASET_DIR.parent.parent)}`")
    st.button("➕ רשומה חדשה", on_click=on_new, width="stretch")

    cat_filter = st.multiselect("סינון לפי קטגוריה", index.domains + [NO_CATEGORY])
    dim_opts = [f"{d.name}: {v}" for d in index.dimensions for v in d.values]
    dim_filter = st.multiselect("סינון לפי מימד", dim_opts)

    def matches(r: dict) -> bool:
        cat = r["expected"].get("category") or NO_CATEGORY
        if cat_filter and cat not in cat_filter:
            return False
        meta = r.get("metadata") or {}
        return all(v in meta.get(k, []) for k, v in (f.split(": ", 1) for f in dim_filter))

    shown = [r["id"] for r in records if matches(r)]
    if ss.current_id and ss.current_id not in shown and ss.current_id in by_id:
        shown.insert(0, ss.current_id)
    options = [NEW] + shown
    if ss.sel not in options:
        ss.sel = NEW
    st.selectbox(
        f"רשומות ({len(shown)})", options, key="sel", on_change=on_select,
        format_func=lambda rid: "✳️ חדשה" if rid == NEW else f"{rid} · {by_id[rid]['query'][:45]}",
    )

    st.divider()
    st.subheader("📊 כיסוי מימדים")
    cats = Counter(r["expected"].get("category") or NO_CATEGORY for r in records)
    st.markdown("**category**  \n" + "  \n".join(
        (f"{c} — {cats[c]}" if cats[c] else f":red[{c} — 0]") for c in index.domains + [NO_CATEGORY]))
    for d in index.dimensions:
        counts = Counter(v for r in records for v in (r.get("metadata") or {}).get(d.name, []))
        st.markdown(f"**{d.name}**  \n" + "  \n".join(
            (f"{v} — {counts[v]}" if counts[v] else f":red[{v} — 0]") for v in d.values))


# --- field widgets -------------------------------------------------------------
def field_hint(e: dict) -> str:
    bits = [e["hierarchy"], f"type: {e['type']}"]
    if e.get("range"):
        bits.append(f"range: {e['range'].get('min')} – {e['range'].get('max')}")
    for k, v in (e.get("extra") or {}).items():
        bits.append(f"{k}: {', '.join(map(str, v)) if isinstance(v, list) else v}")
    return "  \n".join(bits)


def number_input(col, label: str, value, key: str, as_float: bool):
    if as_float:
        return col.number_input(label, value=None if value is None else float(value), step=1.0, key=key)
    return col.number_input(label, value=None if value is None else int(value), step=1, key=key)


def field_widget(e: dict, default, g: int):
    """Render the input for one inventory field; return its param value or None if unset."""
    name, t = e["field_name"], e["type"]
    key, hint = f"{g}:v:{e['id']}", field_hint(e)

    if t.startswith("enum"):
        opts = e.get("possible_values") or []
        d = default if isinstance(default, list) else ([] if default is None else [default])
        sel = st.multiselect(name, opts, default=[x for x in d if x in opts], key=key, help=hint)
        return None if not sel else (sel[0] if len(sel) == 1 else sel)

    if t in store.NUMERIC_TYPES:
        dmin = dmax = dval = None
        if is_number(default):
            op, dval = "exact", default
        elif isinstance(default, dict):
            dmin, dmax = default.get("min"), default.get("max")
            op = "range" if dmin is not None and dmax is not None else ("min" if dmin is not None else "max")
        else:
            op = "range" if t == "range" else "exact"
        as_float = any(isinstance(v, float) for v in (dmin, dmax, dval))
        c1, c2, c3 = st.columns([2, 1, 1])
        op = c1.radio(name, NUM_OPS, index=NUM_OPS.index(op), horizontal=True, key=f"{key}:op", help=hint)
        if op == "exact":
            return number_input(c2, "ערך", dval, f"{key}:val", as_float)
        lo = number_input(c2, "min", dmin, f"{key}:lo", as_float) if op in ("min", "range") else None
        hi = number_input(c3, "max", dmax, f"{key}:hi", as_float) if op in ("max", "range") else None
        out = {k: v for k, v in (("min", lo), ("max", hi)) if v is not None}
        return out or None

    if t == "boolean":
        idx = None if not isinstance(default, bool) else (0 if default else 1)
        v = st.radio(name, ["true", "false"], index=idx, horizontal=True, key=key, help=hint)
        return None if v is None else v == "true"

    examples = (e.get("extra") or {}).get("examples") or []
    v = st.text_input(name, value=default if isinstance(default, str) else "", key=key, help=hint,
                      placeholder=", ".join(map(str, examples)) if examples else "")
    return v.strip() or None


# --- editor ------------------------------------------------------------------
if "_flash" in ss:
    st.success(ss.pop("_flash"))

g, L = ss.gen, ss.loaded
L_exp = L.get("expected") or {}
L_params = L_exp.get("params") or {}
L_meta = L.get("metadata") or {}

record_id = ss.current_id or store.next_id(records)
st.title(f"✏️ {ss.current_id}" if ss.current_id else f"✳️ רשומה חדשה ({record_id})")

query = st.text_area("שאילתה (query)", value=L.get("query", ""), key=f"{g}:query", height=80)

cat_opts = [NO_CATEGORY] + index.domains
L_cat = L_exp.get("category")
if L_cat is not None and L_cat not in index.domains:
    st.warning(f"Loaded category {L_cat!r} is not in the inventory")
cat_choice = st.selectbox("קטגוריה (category)", cat_opts,
                          index=cat_opts.index(L_cat) if L_cat in cat_opts else 0, key=f"{g}:category")
category = None if cat_choice == NO_CATEGORY else cat_choice

params: dict = {}
if category:
    # Hierarchy: render levels as they become reachable (e.g. יצרנים -> דגמים -> ...).
    picks: dict = {}
    rendered: set[str] = set()
    levels_box = st.container()
    with levels_box:
        st.subheader("היררכיה")
        cols = st.columns(3)
        i = 0
        while True:
            pending = [e for e in index.active_hierarchy(category, picks) if e["field_name"] not in rendered]
            if not pending:
                break
            e = pending[0]
            name = e["field_name"]
            rendered.add(name)
            opts = e.get("possible_values") or []
            dv = L_params.get(name)
            sig = json.dumps(e["conditions"], ensure_ascii=False, sort_keys=True)
            v = cols[i % 3].selectbox(name, opts, index=opts.index(dv) if dv in opts else None,
                                      key=f"{g}:h:{name}:{sig}", placeholder="לא נבחר", help=field_hint(e))
            i += 1
            if v is not None:
                picks[name] = v
                params[name] = v
        if not rendered:
            st.caption("אין היררכיה בקטגוריה זו")

    st.subheader("שדות (params)")
    fields = index.available_fields(category, picks)
    picks_sig = json.dumps(picks, ensure_ascii=False, sort_keys=True)
    chosen = st.multiselect(
        "בחר שדות", list(fields), default=[n for n in L_params if n in fields],
        key=f"{g}:fields:{category}:{picks_sig}",
        format_func=lambda n: f"{n} · {fields[n]['type']}",
    )
    for name in chosen:
        value = field_widget(fields[name], L_params.get(name), g)
        if value is not None:
            params[name] = value
    missing = [n for n in chosen if n not in params]
    if missing:
        st.caption(f"⚠️ שדות ללא ערך (לא יישמרו): {', '.join(missing)}")
    stale = [n for n in L_params if n not in fields and n not in rendered]
    if stale:
        st.warning(f"שדות מהרשומה שאינם זמינים בבחירה הנוכחית: {', '.join(stale)}")

st.subheader("מטא-דאטה — מימדים שהתרחיש מכסה")
metadata: dict = {}
mcols = st.columns(min(3, max(1, len(index.dimensions))))
for i, d in enumerate(index.dimensions):
    loaded_vals = L_meta.get(d.name, [])
    metadata[d.name] = mcols[i % len(mcols)].multiselect(
        d.name, d.values, default=[v for v in loaded_vals if v in d.values], key=f"{g}:m:{d.name}")
    unknown = [v for v in loaded_vals if v not in d.values]
    if unknown:
        mcols[i % len(mcols)].caption(f"⚠️ ערכים לא מוכרים הוסרו: {unknown}")

notes = st.text_area("הערות (notes)", value=L.get("notes", ""), key=f"{g}:notes", height=68)

record = store.canonical({
    "id": record_id, "query": query.strip(),
    "expected": {"category": category, "params": params},
    "metadata": metadata, "notes": notes.strip(),
}, index)
errors, warnings = store.validate(record, index, records)

left, right = st.columns([3, 2])
with right:
    st.caption("תצוגה מקדימה")
    st.json(record, expanded=True)
with left:
    for msg in errors:
        st.error(msg)
    for msg in warnings:
        st.warning(msg)

    b1, b2, b3 = st.columns(3)
    if b1.button("💾 שמירה", type="primary", disabled=bool(errors), width="stretch"):
        store.save(DATA_PATH, store.upsert(records, record), index)
        open_record(record["id"], record)
        finish(record["id"], f"נשמר {record['id']}")
    if b2.button("📄 שכפול", width="stretch"):
        dup = copy.deepcopy(record)
        dup["id"] = None
        open_record(None, dup)
        finish(NEW, "שוכפל לרשומה חדשה — יש לשמור")
    if ss.current_id:
        confirm = b3.checkbox("אישור מחיקה", key=f"{g}:confirm_delete")
        if b3.button("🗑️ מחיקה", disabled=not confirm, width="stretch"):
            store.save(DATA_PATH, [r for r in records if r["id"] != ss.current_id], index)
            deleted = ss.current_id
            open_record(None)
            finish(NEW, f"נמחק {deleted}")
