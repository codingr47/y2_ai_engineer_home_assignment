"""Evidence for docs/taxonomy_point_range_inconsistency.md (read-only; prints JSON).

    python docs/point_range_evidence.py

Reads yad2_search_taxonomy.json and training/data/split1/{train,val,test}.jsonl.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TWINS = {"נדל״ן": {"מחיר": "טווח_מחיר", "קומה": "טווח_קומה", "מס׳_חדרים": "טווח_חדרים", "מ״ר_בנוי": "טווח_מ״ר"},
         "רכב": {"מחיר": "טווח_מחיר", "שנה": "טווח_שנה", "ק״מ": "טווח_ק״מ"}}
NUMERIC_OPS = ("exact", "min", "max", "range")


def rows():
    for split in ("train", "val", "test"):
        with open(REPO / "training" / "data" / "split1" / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["_split"] = split
                yield r


def main():
    tax = json.loads((REPO / "yad2_search_taxonomy.json").read_text(encoding="utf-8"))["קטגוריות"]
    data = list(rows())
    twin_names = {(d, x) for d, m in TWINS.items() for p, r in m.items() for x in (p, r)}

    # taxonomy definitions of the twins and of every numeric point field
    defs = {}
    for d in TWINS:
        gen, flt = tax[d]["מאפיינים_כלליים"], tax[d]["פילטרים_נוספים"]
        defs[d] = {"twins": {p: {"point_spec": gen.get(p), "range_name": r, "range_spec": flt.get(r)}
                             for p, r in TWINS[d].items()},
                   "numeric_point_fields_without_twin": sorted(
                       k for k, v in gen.items() if isinstance(v, dict) and v.get("טיפוס") == "מספר"
                       and k not in TWINS[d]),
                   "range_filters": list(flt)}

    # how the labels use each pair, by quality tier
    use = defaultdict(Counter)
    shapes = defaultdict(Counter)
    for r in data:
        d = r.get("domain")
        if d not in TWINS:
            continue
        sp, params = r.get("spans") or {}, (r.get("expected") or {}).get("params") or {}
        for p, rg in TWINS[d].items():
            for key, kind in ((p, "point"), (rg, "range")):
                if key in sp:
                    op = sp[key][0]["op"]
                    bucket = "exact" if op == "exact" else "bounded"
                    for q in ("all", r["quality"]):
                        use[f"{d}/{p}|{q}"][f"{bucket}->{kind}"] += 1
                    v = params.get(key)
                    shapes[f"{d}/{key}"][f"{op}:{'object' if isinstance(v, dict) else type(v).__name__}"] += 1

    totals = defaultdict(Counter)
    for k, c in use.items():
        q = k.split("|")[1]
        totals[q].update(c)

    # numeric fields WITHOUT a twin: how often their point field carries a bound
    non = defaultdict(Counter)
    for r in data:
        d = r.get("domain")
        for k, sp in (r.get("spans") or {}).items():
            if sp[0]["op"] in NUMERIC_OPS and (d, k) not in twin_names:
                non[f"{d}/{k}"][sp[0]["op"]] += 1
    non_twin = {k: {"n": sum(c.values()), "bounded_share": round(1 - c["exact"] / sum(c.values()), 3)}
                for k, c in sorted(non.items(), key=lambda kv: -sum(kv[1].values()))}

    # car year concepts
    years = Counter()
    for r in data:
        if r.get("domain") == "רכב":
            sp, params = r.get("spans") or {}, (r.get("expected") or {}).get("params") or {}
            for k in ("שנה", "טווח_שנה", "שנות_ייצור"):
                if k in params:
                    op = sp.get(k, [{}])[0].get("op")
                    years[f"{k}|{'exact' if op == 'exact' else 'bounded'}|maker={'יצרנים' in params}"] += 1

    print(json.dumps({"taxonomy": defs, "label_use": {k: dict(v) for k, v in sorted(use.items())},
                      "label_use_totals": {k: dict(v) for k, v in totals.items()},
                      "value_shapes": {k: dict(v) for k, v in sorted(shapes.items())},
                      "numeric_fields_without_twin": non_twin, "car_year_fields": dict(years)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
