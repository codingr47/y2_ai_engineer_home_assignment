"""Answer a recognised search that has no filters ("מחפש בית יפה" -> נדל״ן, {}) instead of escalating it.

The local parsers escalate every answer with empty params (reason `empty_extraction`) before the gate sees it. On the
test set that withholds 55 correct answers out of 62, and Luna gets fewer of them right than the tagger (43). This
wrapper catches that one reason and applies the same gate as every other local answer: the category with `{}` is
returned when the gate accepts its score, and escalates otherwise. In the local-only profile (threshold 0) all are
returned.

It wraps the local parser from outside, because the parsers' own source is bound into the gate artifacts. The
candidate is recomputed (one more inference, on about 3% of queries), since the parser keeps it only for the
duration of its own call.
"""
import json
import logging

import anyio

from src.contracts.models import ParseResponse
from src.frozen_fusion.local_parser import public_payload
from src.llm.injection_guard import PromptInjectionDetected, check_query
from src.parsing.parser import FallbackRequired

logger = logging.getLogger(__name__)


class EmptyParamsParser:
    def __init__(self, local, metrics=None):
        self.local, self.metrics = local, metrics

    async def parse(self, query):
        try:
            return await self.local.parse(query)
        except FallbackRequired as exc:
            score = exc.detail.get("confidence")
            if exc.detail.get("reason") != "empty_extraction" or not self.local.gate.accepts(score):
                if exc.detail.get("reason") == "empty_extraction":
                    self._count("fallback")
                raise
            escalation = exc
        try:
            # Injection text must not be answered: before this wrapper, these queries always reached the LLM path and
            # its guard (HTTP 400). Keep the original escalation instead of returning the attack's category.
            check_query(query)
        except PromptInjectionDetected:
            self._count("fallback")
            raise escalation from None
        candidate = await anyio.to_thread.run_sync(lambda: self.local.candidate(query), limiter=self.local.limiter)
        payload = candidate.payload
        if candidate.reason != "empty_extraction" or not payload or payload.get("params"):
            # not reproducible (should not happen: inference is deterministic): keep the original escalation
            self._count("fallback")
            raise FallbackRequired("empty_extraction", score, self.local.gate.threshold)
        self._count("local")
        logger.info(json.dumps({"event": "parse_decision", "destination": "local", "reason": "empty_params",
                                "confidence": score, "threshold": self.local.gate.threshold}))
        return ParseResponse(**public_payload(payload), confidence=score)

    def _count(self, destination):
        if self.metrics:
            self.metrics.empty_params.labels(destination).inc()
