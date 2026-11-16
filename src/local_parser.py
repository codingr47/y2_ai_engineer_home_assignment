"""Classification, value extraction, output validation, and selective fallback."""
import json
import math
import logging
from dataclasses import dataclass

import anyio

from src.classifier import FusionClassifier
from src.confidence import ConfidenceGate, bindings, features
from src.extraction.extractor import FieldExtractor
from src.models import ParseResponse
from src.parser import FallbackRequired

VEHICLE_KEYS = {"יצרנים": "יצרן", "דגמים": "דגם"}
logger = logging.getLogger("uvicorn.error")


@dataclass
class Candidate:
    payload: dict | None
    vector: object
    reason: str | None


def validate_output(payload, extractor):
    """Validate field names and value shapes against the versioned extraction schema."""
    domain = payload["category"]
    if domain not in {"נדל״ן", "רכב", "יד_שנייה"}:
        raise ValueError("unsupported category")
    for name, value in payload["params"].items():
        fi = extractor.model.fields.get(domain + "/" + name)
        if fi is None:
            raise ValueError("unknown field")
        def numeric(x):
            return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        if fi.type == "numeric":
            if isinstance(value, dict):
                if not value or not set(value) <= {"min", "max"} or not all(numeric(v) for v in value.values()):
                    raise ValueError("invalid numeric bounds")
                if "min" in value and "max" in value and value["min"] > value["max"]:
                    raise ValueError("reversed numeric bounds")
            elif not numeric(value):
                raise ValueError("invalid numeric value")
        elif fi.type == "bool":
            if not isinstance(value, bool):
                raise ValueError("invalid boolean")
        else:
            values = value if isinstance(value, list) else [value]
            if not values or not all(isinstance(v, str) and v.strip() for v in values):
                raise ValueError("invalid text value")
            if fi.type == "enum":
                allowed = set()
                for counts in fi.surfaces.values():
                    for canonical in counts:
                        v = json.loads(canonical)
                        allowed.update(v if isinstance(v, list) else [v])
                if any(v not in allowed for v in values):
                    raise ValueError("unknown enum value")


def public_payload(payload):
    if payload["category"] != "רכב":
        return payload
    return {**payload, "params": {VEHICLE_KEYS.get(k, k): v for k, v in payload["params"].items()}}


class LocalParser:
    def __init__(self, classifier, extractor, gate, fallback_parser=None, metrics=None):
        self.classifier, self.extractor, self.gate = classifier, extractor, gate
        self.fallback_parser, self.metrics = fallback_parser, metrics
        self.limiter = anyio.CapacityLimiter(1)

    @classmethod
    def load(cls, settings, fallback_parser=None, metrics=None):
        classifier = FusionClassifier.load(settings.classifier_artifact_path, settings.e5_model_path)
        extractor = FieldExtractor.load(use_tagger=settings.extractor_use_tagger)
        bound = bindings(settings.classifier_artifact_path, settings.e5_model_path, settings.extractor_use_tagger)
        gate = ConfidenceGate.load(settings.confidence_artifact_path, bound, settings.parse_confidence_threshold)
        classifier.classify("דירה")  # warm the encoder and classifier before serving
        return cls(classifier, extractor, gate, fallback_parser, metrics)

    def candidate(self, query):
        c = self.classifier.classify(query)
        out = self.extractor.extract_all(query, c.domain, c.fields)
        vector = features(c, out)
        payload = self.extractor.to_json(c.domain, out)
        if payload is None or c.domain not in {"נדל״ן", "רכב", "יד_שנייה"}:
            return Candidate(None, vector, "unsupported_category")
        try:
            validate_output(payload, self.extractor)
        except ValueError:
            return Candidate(None, vector, "invalid_output")
        if not payload["params"]:
            return Candidate(payload, vector, "empty_extraction")
        return Candidate(payload, vector, None)

    async def parse(self, query):
        def infer():
            candidate = self.candidate(query)
            return candidate, self.gate.score(candidate.vector)
        candidate, score = await anyio.to_thread.run_sync(infer, limiter=self.limiter)
        if self.metrics and score is not None:
            self.metrics.confidence.observe(score)
        reason = candidate.reason
        if reason is None and not self.gate.accepts(score):
            reason = "confidence_unavailable" if not self.gate.enabled or score is None else "low_confidence"
        if reason is None:
            logger.info(json.dumps({"event": "parse_decision", "destination": "local",
                                    "reason": "accepted", "confidence": score, "threshold": self.gate.threshold}))
            if self.metrics:
                self.metrics.routing.labels("local", "accepted").inc()
            return ParseResponse(**public_payload(candidate.payload), confidence=score)
        if self.metrics:
            self.metrics.routing.labels("fallback", reason).inc()
        logger.info(json.dumps({"event": "parse_decision", "destination": "fallback", "reason": reason,
                                "confidence": score, "threshold": self.gate.threshold}))
        if self.fallback_parser is None:
            raise FallbackRequired(reason, score, self.gate.threshold)
        response = ParseResponse.model_validate(await self.fallback_parser.parse(query))
        params = response.params
        if response.category == "רכב":
            if any(k in VEHICLE_KEYS for k in params):
                raise ValueError("fallback must use public vehicle keys")
            reverse = {v: k for k, v in VEHICLE_KEYS.items()}
            params = {reverse.get(k, k): v for k, v in params.items()}
        validate_output({"category": response.category, "params": params}, self.extractor)
        return response
