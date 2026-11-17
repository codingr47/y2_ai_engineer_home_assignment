"""Turn split1 jsonl rows into classifier labels and split val 50/50 into val_calib / val_eval.

Label record (artifacts/labels/<split>.json), exactly:
    {"id": "syn_0023", "domain_label": "יד_שנייה",
     "field_labels": ["יד_שנייה/סקטורים", "יד_שנייה/תתי_קטגוריות", "יד_שנייה/מחיר"]}
Side info needed for embedding / grouping / breakdowns goes to <split>_meta.json (same order).
"""
from __future__ import annotations

import json
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold

from config import DATA_DIR, LABELS_DIR, NONE_DOMAIN, SEED


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_label(row: dict) -> dict:
    domain = row.get("domain")
    if domain is None:
        return {"id": row["id"], "domain_label": NONE_DOMAIN, "field_labels": []}
    params = (row.get("expected") or {}).get("params") or {}
    return {"id": row["id"], "domain_label": domain, "field_labels": [f"{domain}/{k}" for k in params]}


def to_meta(row: dict) -> dict:
    md = row.get("metadata") or {}
    return {
        "id": row["id"],
        "query": row["query"],
        "family": row.get("family") or row["id"],
        "quality": row.get("quality"),
        "language": (md.get("language") or ["?"])[0],
        "complexity": (md.get("complexity") or ["?"])[0],
    }


def split_val(rows: list[dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """50/50, grouped by family (rephrasings never straddle), stratified by domain."""
    y = [to_label(r)["domain_label"] for r in rows]
    groups = [r.get("family") or r["id"] for r in rows]
    best = None
    for s in range(seed, seed + 20):  # deterministic: keep the most domain-balanced of 20 seeded splits
        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=s)
        a, b = next(sgkf.split(rows, y, groups))
        ca, cb = Counter(y[i] for i in a), Counter(y[i] for i in b)
        gap = max(abs(ca[k] - cb[k]) for k in set(y))
        if best is None or gap < best[0]:
            best = (gap, a, b)
    _, calib_idx, eval_idx = best
    return [rows[i] for i in calib_idx], [rows[i] for i in eval_idx]


def _dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    raw = {s: read_jsonl(DATA_DIR / f"{s}.jsonl") for s in ("train", "val", "test")}
    val_calib, val_eval = split_val(raw["val"])
    splits = {"train": raw["train"], "val_calib": val_calib, "val_eval": val_eval, "test": raw["test"]}

    vocab = sorted({f for r in raw["train"] for f in to_label(r)["field_labels"]})
    for name, rows in splits.items():
        unseen = {f for r in rows for f in to_label(r)["field_labels"]} - set(vocab)
        assert not unseen, f"{name} has labels unseen in train: {unseen}"
        _dump([to_label(r) for r in rows], LABELS_DIR / f"{name}.json")
        _dump([to_meta(r) for r in rows], LABELS_DIR / f"{name}_meta.json")

    _dump(vocab, LABELS_DIR / "field_vocab.json")
    _dump({"val_calib": [r["id"] for r in val_calib], "val_eval": [r["id"] for r in val_eval]},
          LABELS_DIR / "val_split.json")

    fam = {n: {r.get("family") or r["id"] for r in rows} for n, rows in splits.items()}
    assert not fam["val_calib"] & fam["val_eval"], "family leak between val_calib and val_eval"
    print(f"field vocab: {len(vocab)}")
    for name, rows in splits.items():
        dom = Counter(to_label(r)["domain_label"] for r in rows)
        print(f"{name:10s} rows={len(rows):5d} families={len(fam[name]):5d} {dict(dom)}")


if __name__ == "__main__":
    main()
