import copy

import pytest

from scripts.evaluation_cost import estimate_cost
from scripts.evaluation_cost import request_cost


def summary(tokens=None, model="gpt-6-luna"):
    return {"configuration": {"model": model}, "overall": {"rows": 10, "failures": {},
            "tokens": tokens if tokens is not None else {"input": 1000000, "cached_input": 400000,
                "cache_write_30m": 100000, "cache_write_12h": 50000, "output": 200000, "reasoning": 100000}}}


def test_estimate_accounts_for_disjoint_input_buckets_and_reasoning_once():
    result = estimate_cost(summary())
    assert result["estimated_total_usd"] == pytest.approx(.174)
    assert result["billable_tokens"]["input"] == 450000
    assert result["estimated_per_query_usd"] == pytest.approx(.0174)
    assert result["projected_10m_queries_usd"] == pytest.approx(174000)
    assert result["price_version"] == "2026-11-17"
    assert len(result["price_table_sha256"]) == 64
    assert "reasoning" not in result["breakdown_usd"]


@pytest.mark.parametrize("tokens,reason", [({}, "token_usage_not_reported"),
                                           ({"input": 1, "cached_input": 2, "output": 0}, "invalid_token_usage"),
                                           ({"input": -1, "output": 1}, "invalid_token_usage"),
                                           ({"input": float("nan"), "output": 1}, "invalid_token_usage")])
def test_missing_or_invalid_usage_is_unavailable(tokens, reason):
    result = estimate_cost(summary(tokens))
    assert result["estimated_total_usd"] is None and result["reason"] == reason


def test_unknown_models_do_not_silently_get_a_price():
    result = estimate_cost(summary(model="unknown-model"))
    assert result["status"] == "unavailable" and result["estimated_total_usd"] is None


def test_per_request_cost_handles_reported_usage_free_cache_hits_and_unknown_failures():
    priced = request_cost("gpt-6-luna", {"usage": {"input": 1000, "output": 200, "reasoning": 100}, "provider_called": True})
    assert priced["estimated_cost_usd"] == pytest.approx(.0002)
    assert request_cost("gpt-6-luna", {"usage": {}, "provider_called": False})["estimated_cost_usd"] == 0
    unknown = request_cost("gpt-6-luna", {"usage": {}, "provider_called": True, "error": "http_timeout"})
    assert unknown["estimated_cost_usd"] is None and unknown["cost_status"] == "unavailable"


def test_reported_failures_are_included_and_marked():
    data = summary()
    data["overall"]["failures"] = {"http_502": 1}
    before = copy.deepcopy(data)
    result = estimate_cost(data)
    assert result["has_failed_requests"] and result["reported_usage_only"]
    assert result["estimated_total_usd"] == pytest.approx(.174)
    assert data == before
