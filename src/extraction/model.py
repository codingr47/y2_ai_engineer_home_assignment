"""The extractor's field model: what each field looks like, learned from training rows + the taxonomy.

Built once by training_frozen_fusion/experiment_4_extraction/build.py and saved as JSON (artifacts/lexicon_v2.json):
- type per field: numeric | bool | date | text (open vocabulary) | enum
- surfaces: folded surface form -> {canonical value (JSON): count}   (enum / text / bool feature words)
- cues (numeric): token -> count of appearing inside / just before the field's gold spans
- value range (numeric): mean / std of log10(|value|) for plausibility
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .aliases import ALIASES
from .numbers import _token_value
from .text import NEGATIONS, fold, prefix_variants, tokens

DATE_FIELDS = {"תאריך_כניסה", "טסט_עד"}
OPEN_TEXT_FIELDS = {"רחוב", "שכונה", "מותג"}
LOCATION_FIELDS = {"עיר", "שכונה", "רחוב", "אזור"}


@dataclass
class FieldInfo:
    name: str                                   # domain-qualified
    type: str
    surfaces: dict[str, dict[str, int]] = field(default_factory=dict)
    cues: dict[str, int] = field(default_factory=dict)
    log_mu: float = 0.0
    log_sigma: float = 1.0
    list_ok: bool = False
    n_train: int = 0


@dataclass
class ExtractorModel:
    schema: str
    fields: dict[str, FieldInfo]
    cue_totals: dict[str, dict[str, int]]      # domain -> token -> count over all numeric fields
    common_tokens: list[str] = field(default_factory=list)   # frequent query words (stop open-text capture)
    unit_tokens: list[str] = field(default_factory=list)     # words that follow numbers inside numeric spans

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, rows: list[dict], taxonomy: dict, schema: str) -> "ExtractorModel":
        types = defaultdict(Counter)
        surfaces = defaultdict(lambda: defaultdict(Counter))
        cues = defaultdict(Counter)
        inside = Counter()  # non-number words inside numeric spans (units / bound words), for span extension
        logs = defaultdict(list)
        lists = set()
        for r in rows:
            dom = r.get("domain")
            if dom is None:
                continue
            params = (r.get("expected") or {}).get("params") or {}
            spans = r.get("spans") or {}
            q = r["query"]
            qtok = tokens(q)
            for k, v in params.items():
                f = f"{dom}/{k}"
                types[f][_value_type(k, v)] += 1
                if isinstance(v, list):
                    lists.add(f)
                for sp in spans.get(k, []):
                    sv = sp.get("value", v)
                    t = _value_type(k, v)
                    if t == "numeric":
                        for x in (sv.values() if isinstance(sv, dict) else [sv]):
                            if isinstance(x, (int, float)) and x > 0:
                                logs[f].append(math.log10(x))
                        for tk in _cue_tokens(qtok, sp["start"], sp["end"]):
                            cues[f][tk] += 1
                        for tk in _cue_tokens(qtok, sp["start"], sp["end"], left=0):
                            inside[tk] += 1
                    elif t == "bool":
                        feat = _strip_negation(sp["text"])
                        if feat:
                            surfaces[f][feat][json.dumps(True)] += 1
                    else:
                        surfaces[f][fold(sp["text"])][json.dumps(sv, ensure_ascii=False)] += 1
        _add_taxonomy_surfaces(surfaces, taxonomy)
        for f, table in ALIASES.items():
            for surface, canonical in table.items():
                surfaces[f][fold(surface)][json.dumps(canonical, ensure_ascii=False)] += 1
        word_freq = Counter(fold(t.text) for r in rows for t in tokens(r["query"]))
        value_words = {w for f in surfaces for s in surfaces[f] for w in s.split()}
        common = [w for w, _ in word_freq.most_common(400) if w not in value_words or word_freq[w] > 150]
        corrections = {fold(w) for w in ("רגע", "בעצם", "סליחה", "כלומר", "טעות", "כוונתי", "התכוונתי")}
        unit_tokens = [w for w, c in inside.items() if c >= 5 and w not in corrections]
        fields = {}
        for f, c in types.items():
            t = c.most_common(1)[0][0]
            mu, sd = (0.0, 1.0)
            if logs[f]:
                mu = sum(logs[f]) / len(logs[f])
                sd = max(0.25, (sum((x - mu) ** 2 for x in logs[f]) / len(logs[f])) ** 0.5)
            fields[f] = FieldInfo(f, t, {s: dict(v) for s, v in surfaces[f].items()}, dict(cues[f]), mu, sd,
                                  f in lists, sum(c.values()))
        totals = defaultdict(Counter)
        for f, fi in fields.items():
            if fi.type == "numeric":
                totals[f.split("/", 1)[0]].update(fi.cues)
        return cls(schema, fields, {d: dict(c) for d, c in totals.items()}, sorted(common), sorted(unit_tokens))

    # ------------------------------------------------------------------ io
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema": self.schema, "cue_totals": self.cue_totals, "common_tokens": self.common_tokens,
                "unit_tokens": self.unit_tokens,
                "fields": {f: vars(fi) for f, fi in sorted(self.fields.items())}}
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ExtractorModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["schema"], {f: FieldInfo(**fi) for f, fi in d["fields"].items()}, d["cue_totals"],
                   d.get("common_tokens", []), d.get("unit_tokens", []))


def _value_type(name: str, v) -> str:
    if name in DATE_FIELDS:
        return "date"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float, dict)):
        return "numeric"
    if name in OPEN_TEXT_FIELDS:
        return "text"
    return "enum"


def _cue_tokens(qtok, start, end, left: int = 2) -> list[str]:
    """Non-number tokens inside the gold span plus the `left` tokens just before it (prefix-stripped, folded)."""
    inside = [i for i, t in enumerate(qtok) if t.start >= start and t.end <= end]
    if not inside:
        return []
    window = list(range(max(0, inside[0] - left), inside[0])) + inside
    out = []
    for i in window:
        tok = qtok[i].text
        if _token_value(tok) is not None and _token_value(tok)[0] in ("num", "half", "teen", "hund"):
            continue
        out.append(fold(prefix_variants(tok)[-1]) if len(prefix_variants(tok)[-1]) > 2 else fold(tok))
    return out


def _strip_negation(text: str) -> str:
    words = [t.text for t in tokens(text) if not any(v in NEGATIONS for v in prefix_variants(t.text))]
    return fold(" ".join(words))


def _add_taxonomy_surfaces(surfaces, taxonomy: dict):
    """Canonical values from the taxonomy are always valid surfaces of themselves (+ model -> manufacturer)."""
    cats = taxonomy.get("קטגוריות", {})

    def add(f, value, surface=None):
        surfaces[f][fold(surface or value)][json.dumps(value, ensure_ascii=False)] += 1

    for dom, body in cats.items():
        for group in ("מאפיינים_כלליים",):
            for name, spec in (body.get(group) or {}).items():
                if isinstance(spec, list):
                    for v in spec:
                        add(f"{dom}/{name}", v)
                elif isinstance(spec, dict):
                    for v in spec.get("דוגמאות", []) or []:
                        if isinstance(v, str):
                            add(f"{dom}/{name}", v)
        for name in ("מצבי_עסקה", "סוגי_נכס", "סוגי_רכב"):
            for v in body.get(name, []) or []:
                add(f"{dom}/{name}", v)
    for maker, info in (cats.get("רכב", {}).get("יצרנים") or {}).items():
        add("רכב/יצרנים", maker)
        for model in info.get("דגמים", []) or []:
            add("רכב/דגמים", model)
            add("רכב/יצרנים", maker, model)  # a model name implies its manufacturer
        for model, subs in (info.get("תתי_דגמים") or {}).items():
            for s in subs:
                add("רכב/תתי_דגמים", s)
    for sector, body in (cats.get("יד_שנייה", {}).get("סקטורים") or {}).items():
        add("יד_שנייה/סקטורים", sector)
        for sub, attrs in (body.get("תתי_קטגוריות") or {}).items():
            add("יד_שנייה/תתי_קטגוריות", sub)
            for name, spec in (attrs or {}).items():
                if isinstance(spec, list):
                    for v in spec:
                        add(f"יד_שנייה/{name}", v)
