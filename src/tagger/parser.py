"""The fine-tuned tagger as the API's local parser.

    query ─► QueryCleaner ─► 3-seed DictaBERT ensemble ─► SpanNormalizer (+ extractor fallback) ─► payload
       └────► Frozen Fusion (raw query): its parse and gate score, a second opinion ──────────────┐
                                                        gate features ─► TaggerGate ─► local │ LLM fallback

Routing, logging, metrics and the LLM fallback are LocalParser.parse, unchanged: this class only replaces
`candidate()` and the gate. Everything is loaded from versioned artifacts (src/artifacts/tagger) and provisioned
weights (models/fine-tuned-tagger, see scripts/provision_tagger.py); nothing reads training data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.frozen_fusion.confidence import bindings as ff_bindings
from src.frozen_fusion.local_parser import Candidate, LocalParser, public_payload, validate_output

from .clean import QueryCleaner
from .gate import TaggerGate, bindings, features
from .model import TaggerModel
from .normalize import SpanNormalizer, hybrid_vehicle_type
from .predict import predict, to_json
from .vocab import Vocab

SUPPORTED = {"נדל״ן", "רכב", "יד_שנייה"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_members(artifact_dir: Path, model_dir: Path, device) -> tuple[list[TaggerModel], dict]:
    """The ensemble members listed in members.json, each verified by SHA-256 before loading."""
    spec = json.loads((Path(artifact_dir) / "members.json").read_text(encoding="utf-8"))
    vocab = Vocab.from_json(json.loads((Path(artifact_dir) / "vocab.json").read_text(encoding="utf-8")))
    models = []
    for m in spec["members"]:
        path = Path(model_dir) / m["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run python -m scripts.provision_tagger")
        if sha256(path) != m["sha256"]:
            raise ValueError(f"{path.name} does not match members.json")
        model = TaggerModel(str(Path(model_dir) / "encoder"), vocab, pretrained=False)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
        models.append(model.to(device).eval())
    return models, spec


class TaggerParser(LocalParser):
    def __init__(self, models, tokenizer, cleaner, normalizer, ff_parser, gate, spec, device,
                 fallback_parser=None, metrics=None, hybrid_vehicle_type: bool = False):
        super().__init__(ff_parser.classifier, normalizer.ex, gate, fallback_parser, metrics)
        self.models, self.tok, self.cleaner, self.norm = models, tokenizer, cleaner, normalizer
        self.ff, self.spec, self.device, self.hybrid = ff_parser, spec, device, hybrid_vehicle_type

    @classmethod
    def load(cls, settings, fallback_parser=None, metrics=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ff = LocalParser.load(settings)          # Frozen Fusion: its classifier, extractor and audited gate
        models, spec = load_members(settings.tagger_artifact_dir, settings.tagger_model_dir, device)
        tok = AutoTokenizer.from_pretrained(str(Path(settings.tagger_model_dir) / "encoder"))
        cleaner = QueryCleaner.from_file(Path(settings.tagger_artifact_dir) / "clean_vocab.json")
        norm = SpanNormalizer.from_artifacts(settings.tagger_artifact_dir)
        bound = bindings(settings.tagger_artifact_dir, settings.tagger_model_dir,
                         ff_bindings(settings.classifier_artifact_path, settings.e5_model_path,
                                     settings.extractor_use_tagger), settings.confidence_artifact_path)
        gate = TaggerGate.load(settings.tagger_gate_path, bound, settings.parse_confidence_threshold)
        parser = cls(models, tok, cleaner, norm, ff, gate, spec, device, fallback_parser, metrics,
                     settings.tagger_hybrid_vehicle_type)
        parser.candidate("דירה")                # warm both encoders before serving
        return parser

    def candidate(self, query):
        cleaned = self.cleaner.clean(query)
        decode = self.spec["decode"]
        raw = predict(self.models, self.tok, [cleaned], self.device, amp=self.device.type == "cuda")[0]
        pj = to_json(raw, cleaned, self.norm, decode["policy"], decode["presence_threshold"],
                     fallback=decode["fallback"])
        ff = self.ff.candidate(query)
        rec = {"raw": raw, "pred": pj["params"] if pj else None, "pred_domain": raw["domain"], "query": cleaned}
        vector = features(rec, {"pred": public_payload(ff.payload) if ff.payload else None,
                                "score": self.ff.gate.score(ff.vector)})
        if pj is None or pj["category"] not in SUPPORTED:
            return Candidate(None, vector, "unsupported_category")
        if self.hybrid:   # product rule, applied after scoring: the gate was fitted without it
            pj = {**pj, "params": hybrid_vehicle_type(pj["category"], pj["params"])}
        try:
            validate_output(pj, self.extractor)
        except ValueError:
            return Candidate(None, vector, "invalid_output")
        if not pj["params"]:
            return Candidate(pj, vector, "empty_extraction")
        return Candidate(pj, vector, None)
