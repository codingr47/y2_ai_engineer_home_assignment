"""Score a finished label audit: label error rate per pool and overall, with 95% Wilson intervals.

    python -m training_bert.experiment3.audit.score_audit [path/to/audit_verdicts.json]

"unsure" rows are reported separately and counted two ways: as correct (optimistic) and as wrong (pessimistic).
Overall = pool rates weighted by pool size in val (silver_bulk 2,041, synthetic_v2 1,395).
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOL_SIZE = {"silver_bulk": 2041, "synthetic_v2": 1395}


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main(path: Path):
    v = json.loads(path.read_text(encoding="utf-8"))
    done = [x for x in v if x["verdict"]]
    out = {"reviewed": len(done), "of": len(v), "pools": {}}
    for pool in POOL_SIZE:
        xs = [x for x in done if x["pool"] == pool]
        c = Counter(x["verdict"] for x in xs)
        n = len(xs)
        lo, hi = wilson(c["wrong"], n)
        out["pools"][pool] = {"n": n, **c, "error_rate": c["wrong"] / n if n else None, "error_ci95": [lo, hi],
                              "error_rate_if_unsure_wrong": (c["wrong"] + c["unsure"]) / n if n else None}
    tot = sum(POOL_SIZE.values())
    rates = [(POOL_SIZE[p], out["pools"][p]["error_rate"]) for p in POOL_SIZE if out["pools"][p]["error_rate"] is not None]
    out["weighted_error_rate"] = sum(w * r for w, r in rates) / sum(w for w, _ in rates) if rates else None
    out["notes_of_wrong"] = [{"id": x["id"], "pool": x["pool"], "note": x["note"]} for x in done if x["verdict"] == "wrong"]
    (HERE / "audit_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"reviewed {len(done)}/{len(v)}")
    for p, s in out["pools"].items():
        if s["n"]:
            print(f"{p}: {s['n']} rows | wrong {s.get('wrong', 0)} unsure {s.get('unsure', 0)} | error "
                  f"{100 * s['error_rate']:.1f}% (95% CI {100 * s['error_ci95'][0]:.1f}–{100 * s['error_ci95'][1]:.1f}%)")
    if out["weighted_error_rate"] is not None:
        print(f"val-weighted label error rate: {100 * out['weighted_error_rate']:.1f}%")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "audit_verdicts.json")
