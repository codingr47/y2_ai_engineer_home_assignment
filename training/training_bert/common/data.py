"""Data for the fine-tuned tagger: splits, label vocabularies, char-span -> subword BIO alignment.

The data files in training/training_bert/data are only read. Labels follow schema v2: `normalize_row` renames the range-filter
keys (train/val are already v2; test is normalized here at load).

Split roles:
    train      fit
    val_eval   all selection (LR, epochs, data size, decode thresholds)
    val_calib  reserved for a later confidence gate
    test       one final check; never tuned on
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from src.tagger.vocab import DOMAINS, IMPLIED_MIN_COUNT, OPS, Vocab, decode_tags, encode_inputs  # noqa: F401 (re-exported)
from training.training_frozen_fusion.shared.normalize import normalize_row

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPO = ROOT.parent.parent
FF_VAL_SPLIT = REPO / "training/training_frozen_fusion" / "experiment_1" / "artifacts" / "labels" / "val_split.json"
VAL_SPLIT = ARTIFACTS / "val_split.json"



@lru_cache(maxsize=None)
def _read(name: str) -> tuple[dict, ...]:
    rows = []
    for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = normalize_row(json.loads(line))
            r.setdefault("domain", (r.get("expected") or {}).get("category"))
            rows.append(r)
    return tuple(rows)


def build_val_split(seed: int = 42) -> dict[str, list[str]]:
    """val -> val_calib / val_eval by family. Rows Frozen Fusion already assigned keep their side; the rest are split
    50/50 by family (shuffled with `seed`), so families never straddle the two parts."""
    ff = json.loads(FF_VAL_SPLIT.read_text(encoding="utf-8"))
    side = {i: k for k, ids in ff.items() for i in ids}
    rows = _read("val")
    fam_side = {}
    for r in rows:
        if r["id"] in side:
            prev = fam_side.setdefault(r["family"], side[r["id"]])
            assert prev == side[r["id"]], f"family {r['family']} straddles the Frozen Fusion split"
    rest = sorted({r["family"] for r in rows} - set(fam_side))
    random.Random(seed).shuffle(rest)
    for k, fam in enumerate(rest):
        fam_side[fam] = "val_calib" if k % 2 == 0 else "val_eval"
    out = {"val_calib": [], "val_eval": []}
    for r in rows:
        out[fam_side[r["family"]]].append(r["id"])
    return out


@lru_cache(maxsize=None)
def val_split() -> dict[str, tuple[str, ...]]:
    if not VAL_SPLIT.exists():
        VAL_SPLIT.parent.mkdir(parents=True, exist_ok=True)
        VAL_SPLIT.write_text(json.dumps(build_val_split(), ensure_ascii=False, indent=0), encoding="utf-8")
    return {k: tuple(v) for k, v in json.loads(VAL_SPLIT.read_text(encoding="utf-8")).items()}


def load(split: str) -> list[dict]:
    if split in ("val_calib", "val_eval"):
        keep = set(val_split()[split])
        return [r for r in _read("val") if r["id"] in keep]
    return list(_read(split))


def subset(rows: list[dict], n: int | None, seed: int = 0, quality: str | None = None) -> list[dict]:
    """A family-grouped subset of about n rows (whole families, shuffled with `seed`). Nested: for one seed, a smaller
    n is always a prefix of a larger one, so learning-curve points only add data."""
    if quality:
        rows = [r for r in rows if r.get("quality") == quality]
    if n is None or n >= len(rows):
        return list(rows)
    fams = defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r)
    order = sorted(fams)
    random.Random(seed).shuffle(order)
    out = []
    for f in order:
        if len(out) >= n:
            break
        out.extend(fams[f])
    return out


def params(r: dict) -> dict:
    return (r.get("expected") or {}).get("params") or {}


def explicit_spans(r: dict, with_value: bool = False) -> list[tuple]:
    """(start, end, field, op) of the explicit spans, field domain-qualified, sorted by start. Spans whose field is
    not in the row's params are skipped (52 train rows label a span נפח_אחסון while the param is אחסון_GB)."""
    d = r["domain"]
    ps = params(r)
    out = [(s["start"], s["end"], f"{d}/{k}", s["op"]) + ((s.get("value"),) if with_value else ())
           for k, ss in (r.get("spans") or {}).items() for s in ss
           if s.get("origin", "explicit") == "explicit" and k in ps]
    return sorted(out)


def encode(r: dict, tok, vocab: Vocab, max_len: int = 128) -> dict:
    """Tokenize one row and attach labels: domain, field presence, BIO tag per subword, span (token range, op),
    implied value per implied field (-100 = none)."""
    enc = encode_inputs(r["query"], tok, max_len)
    offs = enc["offsets"]
    tag_id = {t: i for i, t in enumerate(vocab.tags)}
    tags = [-100 if s == e else 0 for s, e in offs]
    spans = []
    for s, e, f, op, val in explicit_spans(r, with_value=True):
        if f not in vocab.fields:
            continue
        idx = [i for i, (a, b) in enumerate(offs) if a != b and a < e and b > s]
        if not idx:
            continue
        for j, i in enumerate(idx):
            tags[i] = tag_id[("B-" if j == 0 else "I-") + f]
        spans.append((idx[0], idx[-1] + 1, OPS.index(op), vocab.value_id(f, val)))
    fidx = {f: i for i, f in enumerate(vocab.fields)}
    present = [0.0] * len(vocab.fields)
    implied = {}
    if r["domain"]:
        for k, v in params(r).items():
            f = f"{r['domain']}/{k}"
            if f in fidx:
                present[fidx[f]] = 1.0
            if f in vocab.implied and not (r.get("spans") or {}).get(k) and isinstance(v, str) \
                    and v in vocab.implied[f]:
                implied[f] = vocab.implied[f].index(v)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "offsets": offs,
            "domain": DOMAINS.index(r["domain"] or "none"), "present": present, "tags": tags,
            "spans": spans, "implied": implied}
