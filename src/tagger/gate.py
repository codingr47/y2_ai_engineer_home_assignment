"""Confidence gate of the tagger parser: features, version-bound artifact, accept / route decision.

Features (one vector per query, same code at training and runtime):
- the tagger: domain probability, lowest per-token tag probability, lowest operator probability, ensemble agreement,
  presence margins, field and span counts, whether the extractor fallback added a field, query length, domain
- Frozen Fusion as a second opinion: same JSON / domain / field set as the tagger, its own gate score, parsed at all

The artifact (`tagger-gate-1`) holds a logistic model fitted offline (scripts/build_tagger_gate.py) and the
threshold chosen on calibration data. It is bound to everything that changes the features or their meaning: the
ensemble weights, the tagger artifacts, the encoder config and tokenizer, the runtime code, and Frozen Fusion's
artifacts. A mismatch refuses to load, as the Frozen Fusion gate does.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np

from src.extraction.types import values_equal

DOMS = ("none", "נדל״ן", "רכב", "יד_שנייה")
FEATURE_NAMES = (
    "domain_prob", "token_min", "op_min", "agree_share", "agree_all", "presence_min_in", "presence_max_out",
    "n_fields", "n_spans", "fallback_added", "log_query_len", *(f"domain_{d}" for d in DOMS),
    "ff_same_json", "ff_same_domain", "ff_same_fields", "ff_score", "ff_parsed",
)
SCHEMA = "tagger-gate-1"
VEHICLE_KEYS = {"יצרנים": "יצרן", "דגמים": "דגם"}   # internal -> public, as src.local_parser.public_payload


def json_equal(pred: dict | None, gold: dict | None) -> bool:
    """Exact {category, params} equality with type-aware values; a non-search matches an empty parse."""
    gp = (gold or {}).get("params") or {}
    if gold is None or not gold.get("category"):
        return pred is None or not pred.get("params")
    if pred is None or pred.get("category") != gold.get("category"):
        return False
    pp = pred["params"]
    return set(pp) == set(gp) and all(values_equal(pp[k], v) for k, v in gp.items())


def _public(payload: dict) -> dict:
    if payload["category"] != "רכב":
        return payload
    return {**payload, "params": {VEHICLE_KEYS.get(k, k): v for k, v in payload["params"].items()}}


def tagger_features(rec: dict) -> list[float]:
    """rec: {"raw": predict() output, "pred": params (internal keys) or None, "query": the query the tagger read}."""
    raw = rec["raw"]
    c = raw.get("conf") or {}
    pres = raw.get("present") or {}
    pred = rec.get("pred") or {}
    d = raw["domain"]
    pred_fields = {f"{d}/{k}" for k in pred}
    span_fields = {sp[2] for sp in raw["spans"]}
    p_in = [pres.get(f, 0.0) for f in pred_fields]
    p_out = [p for f, p in pres.items() if f not in pred_fields]
    return [c.get("domain", 1.0), c.get("token_min", 1.0), c.get("op_min", 1.0),
            c.get("agree", 1) / max(1, c.get("members", 1)), float(c.get("agree", 1) == c.get("members", 1)),
            min(p_in) if p_in else 1.0, max(p_out) if p_out else 0.0,
            len(pred), len(raw["spans"]), float(bool(pred_fields - span_fields - set(raw.get("implied") or {}))),
            np.log1p(len(rec["query"])), *[float(d == x) for x in DOMS]]


def tagger_json(rec: dict) -> dict | None:
    if rec["pred_domain"] in (None, "none") or rec["pred"] is None:
        return None
    return _public({"category": rec["pred_domain"], "params": rec["pred"]})


def ff_features(rec: dict, ff: dict) -> list[float]:
    """ff: {"pred": Frozen Fusion's public payload or None, "score": its gate score}."""
    t, f = tagger_json(rec), ff["pred"]
    same = json_equal(f, t) and json_equal(t, f) if (t or f) else True
    same_dom = (t or {}).get("category") == (f or {}).get("category")
    same_fields = same_dom and set((t or {}).get("params") or {}) == set((f or {}).get("params") or {})
    return [float(same), float(same_dom), float(same_fields), ff["score"], float(f is not None)]


def features(rec: dict, ff: dict) -> np.ndarray:
    v = np.array(tagger_features(rec) + ff_features(rec, ff), dtype=float)
    if not np.isfinite(v).all():
        raise ValueError("non-finite gate features")
    return v


TEXT_SUFFIXES = {".py", ".json", ".txt"}


def _digest(paths) -> str:
    """SHA-256 over file names and contents. Text files are hashed with LF line endings, so a checkout that
    converts them to CRLF (git autocrlf on Windows) still matches the artifact built elsewhere."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda p: str(p)):
        p = Path(p)
        h.update(p.name.encode())
        if p.suffix in TEXT_SUFFIXES:
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
            continue
        with open(p, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


ARTIFACT_FILES = ("vocab.json", "members.json", "lexicon_v2.json", "numeric_words.json", "common_words.json",
                  "clean_vocab.json", "taxonomy_v2.json")


ARCH_KEYS = ("model_type", "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
             "intermediate_size", "hidden_act", "max_position_embeddings", "type_vocab_size", "layer_norm_eps",
             "position_embedding_type", "pad_token_id")


def encoder_fingerprint(encoder_dir: Path) -> str:
    """Canonical identity of the encoder: the architecture-defining config fields and the full tokenizer vocabulary.
    Unlike a file hash or the whole config, it is the same under any transformers version (newer versions add
    config defaults) and any line endings, so the Docker image's own transformers install still matches."""
    from transformers import AutoConfig, AutoTokenizer
    full = AutoConfig.from_pretrained(str(encoder_dir)).to_dict()
    cfg = {k: full.get(k) for k in ARCH_KEYS}
    tok = AutoTokenizer.from_pretrained(str(encoder_dir))
    ident = {"config": cfg, "vocab": sorted(tok.get_vocab().items()), "class": type(tok).__name__.replace("Fast", ""),
             "special": sorted((k, str(v)) for k, v in tok.special_tokens_map.items())}
    return hashlib.sha256(json.dumps(ident, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def bindings(artifact_dir: Path, model_dir: Path, ff_bindings: dict, ff_gate_path: Path) -> dict:
    """What the gate's meaning depends on. Member weights are bound through members.json (their SHA-256 list),
    which the loader verifies against the files on disk."""
    src = Path(__file__).resolve().parents[1]
    return {
        "artifacts": _digest([Path(artifact_dir) / f for f in ARTIFACT_FILES]),
        "encoder": encoder_fingerprint(Path(model_dir) / "encoder"),
        "code": _digest([*sorted((src / "tagger").glob("*.py")), *sorted((src / "extraction").glob("*.py"))]),
        "frozen_fusion": {"bindings": ff_bindings, "gate": _digest([Path(ff_gate_path)])},
        "features": hashlib.sha256(json.dumps(FEATURE_NAMES).encode()).hexdigest(),
    }


class TaggerGate:
    def __init__(self, artifact: dict | None = None, threshold: float | None = None):
        self.artifact = artifact
        self.enabled = bool(artifact and artifact.get("acceptance_enabled"))
        self.threshold = threshold if threshold is not None else (artifact or {}).get("threshold")
        if self.threshold is not None and (not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1):
            raise ValueError("invalid confidence threshold")

    @classmethod
    def load(cls, path: Path, expected_bindings: dict, threshold: float | None = None) -> "TaggerGate":
        if not Path(path).exists():
            return cls(threshold=threshold)
        artifact = joblib.load(path)
        if artifact.get("schema") != SCHEMA or artifact.get("bindings") != expected_bindings:
            raise ValueError("tagger gate artifact does not match the active parser")
        if tuple(artifact.get("features", ())) != FEATURE_NAMES:
            raise ValueError("tagger gate feature schema mismatch")
        if artifact.get("acceptance_enabled") and artifact.get("threshold") is None:
            raise ValueError("acceptance-enabled gate artifact requires a threshold")
        return cls(artifact, threshold)

    def score(self, vector) -> float | None:
        if not self.artifact or vector is None:
            return None
        z = (np.asarray(vector, dtype=float) - self.artifact["mu"]) / self.artifact["sd"]
        return float(self.artifact["model"].predict_proba(z[None, :])[0, 1])

    def accepts(self, score) -> bool:
        return self.enabled and score is not None and self.threshold is not None and score >= self.threshold
