"""Reproducible estimates using reported usage and a dated public price snapshot."""
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

PRICES = Path(__file__).resolve().parents[1] / "eval/openai_prices.json"


def request_cost(model, record):
    if record.get("provider_called") is False:
        return {"estimated_cost_usd": 0.0, "cost_status": "no_provider_call"}
    summary = {"configuration": {"model": model}, "overall": {"rows": 1, "tokens": record.get("usage", {}),
               "failures": {record["error"]: 1} if record.get("error") else {}}}
    result = estimate_cost(summary)
    return {"estimated_cost_usd": result["estimated_total_usd"], "cost_status": result["status"],
            "cost_reason": result.get("reason"), "cost_breakdown_usd": result.get("breakdown_usd")}


def estimate_cost(summary, price_path=PRICES):
    source = Path(price_path).read_bytes()
    table = json.loads(source)
    model = summary["configuration"].get("model")
    rates = table["models"].get(model)
    result = {"currency": table["currency"], "model": model, "price_version": table["version"],
              "verified_on": table["verified_on"], "price_table_sha256": hashlib.sha256(source).hexdigest(),
              "service_tier": table["service_tier"], "assumptions": table["assumptions"],
              "source": rates["source"] if rates else table["source"],
              "rates_per_million_tokens": rates, "estimated_total_usd": None}
    if rates is None:
        return {**result, "status": "unavailable", "reason": "model_not_in_price_table"}
    tokens = summary["overall"].get("tokens", {})
    if "input" not in tokens or "output" not in tokens:
        return {**result, "status": "unavailable", "reason": "token_usage_not_reported"}
    try:
        counts = {key: Decimal(str(tokens.get(key, 0))) for key in ("input", "output", "cached_input", "cache_write_30m", "cache_write_12h")}
        if any(not value.is_finite() or value < 0 or value != value.to_integral_value() for value in counts.values()):
            raise ValueError("invalid token count")
        counts["input"] -= counts["cached_input"] + counts["cache_write_30m"] + counts["cache_write_12h"]
        if counts["input"] < 0:
            raise ValueError("input breakdown exceeds total")
    except (InvalidOperation, ValueError):
        return {**result, "status": "unavailable", "reason": "invalid_token_usage"}
    costs = {key: count * Decimal(str(rates[key])) / 1000000 for key, count in counts.items()}
    total = sum(costs.values(), Decimal(0))
    queries = summary["overall"]["rows"]
    average = total / queries if queries else None
    return {**result, "status": "estimated", "reported_usage_only": True,
            "has_failed_requests": bool(summary["overall"].get("failures")),
            "billable_tokens": {key: int(value) for key, value in counts.items()},
            "breakdown_usd": {key: float(value) for key, value in costs.items()},
            "estimated_total_usd": float(total), "estimated_per_query_usd": float(average) if average is not None else None,
            "projected_10m_queries_usd": float(average * 10000000) if average is not None else None,
            "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}
