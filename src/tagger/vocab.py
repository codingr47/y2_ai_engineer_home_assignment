"""Label space of the fine-tuned tagger: domains, span operators, fields, BIO tags, implied and categorical values.

Shared by training (training_bert) and the runtime parser. Pure: no data files are read here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

DOMAINS = ("none", "נדל״ן", "רכב", "יד_שנייה")
OPS = ("value", "exact", "min", "max", "range", "true", "false")
IMPLIED_MIN_COUNT = 20     # a field gets a span-less value head when it is span-less this often in train


def _params(r: dict) -> dict:
    return (r.get("expected") or {}).get("params") or {}


@dataclass
class Vocab:
    fields: tuple[str, ...]                    # domain-qualified
    implied: dict[str, tuple]                  # field -> its span-less values
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
        fields = sorted({f"{r['domain']}/{k}" for r in rows if r["domain"] for k in _params(r)})
        spanless = defaultdict(Counter)
        for r in rows:
            if not r["domain"]:
                continue
            for k, v in _params(r).items():
                if not (r.get("spans") or {}).get(k) and isinstance(v, str):
                    spanless[f"{r['domain']}/{k}"][v] += 1
        implied = {f: tuple(sorted(c)) for f, c in sorted(spanless.items()) if sum(c.values()) >= IMPLIED_MIN_COUNT}
        values = defaultdict(set)
        for r in rows:
            for k, ss in (r.get("spans") or {}).items():
                f = f"{r['domain']}/{k}"
                if enum_fields and f in enum_fields and k in _params(r):
                    values[f].update(s["value"] for s in ss if isinstance(s.get("value"), str))
        return cls(tuple(fields), implied, {f: tuple(sorted(v)) for f, v in sorted(values.items()) if v})


def encode_inputs(query: str, tok, max_len: int = 128) -> dict:
    """Subword ids, attention mask and character offsets of one query (inference: no labels)."""
    enc = tok(query, truncation=True, max_length=max_len, return_offsets_mapping=True)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "offsets": enc["offset_mapping"]}


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
