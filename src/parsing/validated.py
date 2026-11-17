"""Deterministic taxonomy check on local answers, outside the gate-bound local parser code.

The local parsers validate field names and value shapes; this adds the same closed-catalog check the LLM path uses
(enum values from the taxonomy, [when ...] conditions, scalar hierarchy levels). A local answer that fails it is
escalated like any other unsure answer, never returned.
"""
import json
import logging

from src.llm.catalog import VEHICLE_KEYS, Catalog
from src.parsing.parser import FallbackRequired

logger = logging.getLogger(__name__)
PUBLIC_TO_CATALOG = {public: key for key, public in VEHICLE_KEYS.items()}


def catalog_violation(catalog, response):
    """The reason a response breaks the taxonomy, or None when it is valid."""
    params = dict(response.params)
    if response.category == "רכב":
        params = {PUBLIC_TO_CATALOG.get(k, k): v for k, v in params.items()}
    try:
        catalog.validate({"category": response.category, "params": params, "confidence": response.confidence})
    except (ValueError, TypeError, KeyError) as exc:
        return str(exc).splitlines()[0][:80]
    return None


class CatalogCheckedParser:
    """Wraps a local parser. Invalid answers go to `fallback` when given, else raise FallbackRequired so an outer
    FallbackRouter (or the HTTP layer, in local-only mode) handles them like low-confidence answers."""

    def __init__(self, local, catalog=None, metrics=None, fallback=None):
        self.local, self.catalog, self.metrics, self.fallback = local, catalog or Catalog(), metrics, fallback

    async def parse(self, query):
        response = await self.local.parse(query)
        violation = catalog_violation(self.catalog, response)
        if violation is None:
            return response
        if self.metrics:   # parse_routing_total already counted this as local/accepted at the gate
            self.metrics.output_rejections.labels("local").inc()
        logger.info(json.dumps({"event": "parse_decision", "destination": "fallback", "reason": "invalid_output",
                                "violation": violation, "confidence": response.confidence}))
        if self.fallback is None:
            raise FallbackRequired("invalid_output", response.confidence)
        return await self.fallback.parse(query)
