"""Data for the fine-tuned tagger: splits, label vocabularies, char-span -> subword BIO alignment.

The data files in training_bert/data are only read. Labels follow schema v2: `normalize_row` renames the range-filter
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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from training_frozen_fusion.shared.normalize import normalize_row

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPO = ROOT.parent
FF_VAL_SPLIT = REPO / "training_frozen_fusion" / "experiment_1" / "artifacts" / "labels" / "val_split.json"
VAL_SPLIT = ARTIFACTS / "val_split.json"

DOMAINS = ("none", "נדל״ן", "רכב", "יד_שנייה")
OPS = ("value", "exact", "min", "max", "range", "true", "false")
IMPLIED_MIN_COUNT = 20     # a field gets a span-less value head when it is span-less this often in train


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


@dataclass
class Vocab:
    fields: tuple[str, ...]                    # domain-qualified
    implied: dict[str, tuple]                  # field -> its span-less values (JSON strings)
    values: dict[str, tuple] = None            # categorical field -> the span values seen in train (value head)

    def __post_init__(self):
        self.values = self.values or {}
        self.value_index = [(f, v) for f in sorted(self.values) for v in self.values[f]]
        self._vid = {fv: i for i, fv in enumerate(self.value_index)}

    def value_id(self, field: str, value) -> int:
        return self._vid.get((field, value), -100) if isinstance(value, str) else -100

    @property
    def tags(self) -> tuple[str, ...]:
        return ("O",) + tuple(f"{p}-{f}" for f in self.fields for p in ("B", "I"))

    def to_json(self) -> dict:
        return {"fields": list(self.fields), "implied": {k: list(v) for k, v in self.implied.items()},
                "values": {k: list(v) for k, v in self.values.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "Vocab":
        return cls(tuple(d["fields"]), {k: tuple(v) for k, v in d["implied"].items()},
                   {k: tuple(v) for k, v in (d.get("values") or {}).items()})

    @classmethod
    def build(cls, rows: list[dict], enum_fields: set[str] | None = None) -> "Vocab":
        """enum_fields: categorical fields that get a learned value head (None = no value head)."""
        fields = sorted({f"{r['domain']}/{k}" for r in rows if r["domain"] for k in params(r)})
        spanless = defaultdict(Counter)
        for r in rows:
            if not r["domain"]:
                continue
            for k, v in params(r).items():
                if not (r.get("spans") or {}).get(k) and isinstance(v, str):
                    spanless[f"{r['domain']}/{k}"][v] += 1
        implied = {f: tuple(sorted(c)) for f, c in sorted(spanless.items()) if sum(c.values()) >= IMPLIED_MIN_COUNT}
        values = defaultdict(set)
        for r in rows:
            for k, ss in (r.get("spans") or {}).items():
                f = f"{r['domain']}/{k}"
                if enum_fields and f in enum_fields and k in params(r):
                    values[f].update(s["value"] for s in ss if isinstance(s.get("value"), str))
        return cls(tuple(fields), implied, {f: tuple(sorted(v)) for f, v in sorted(values.items()) if v})


def encode(r: dict, tok, vocab: Vocab, max_len: int = 128) -> dict:
    """Tokenize one row and attach labels: domain, field presence, BIO tag per subword, span (token range, op),
    implied value per implied field (-100 = none)."""
    enc = tok(r["query"], truncation=True, max_length=max_len, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
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


def decode_tags(tag_ids: list[int], offsets: list[tuple[int, int]], vocab: Vocab) -> list[tuple[int, int, str, int, int]]:
    """BIO tag ids -> (char_start, char_end, field, tok_start, tok_end). An I- tag continuing a different field, or an
    I- with no B-, starts a new span (lenient decoding)."""
    tags = vocab.tags
    out, cur = [], None
    for i, (t, (a, b)) in enumerate(zip(tag_ids, offsets)):
        if a == b:
            continue
        name = tags[t] if t >= 0 else "O"
        if name == "O":
            cur = None
            continue
        p, f = name.split("-", 1)
        if p == "I" and cur is not None and cur[2] == f:
            cur[1], cur[4] = b, i + 1
        else:
            cur = [a, b, f, i, i + 1]
            out.append(cur)
    return [tuple(c) for c in out]
