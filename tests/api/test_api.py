"""HTTP contract, instrumentation, and cache integration checks."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.app import create_app
from src.infrastructure.cache import LRUCache
from src.contracts.models import ParseResponse
from src.parsing.parser import StubParser


class CountingParser(StubParser):
    def __init__(self):
        self.calls = 0

    async def parse(self, query):
        self.calls += 1
        return await super().parse(query)


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


@pytest.mark.parametrize("query", [
    "דירת 3 חדרים בירושלים עד מליון שח",
    "טויוטה קורולה 2018-2021 עד 70 אלף שח צבע לבן",
])
def test_fixed_response(client, query):
    response = client.post("/parse", json={"q": query})
    assert response.status_code == 200
    assert response.json() == {
        "category": "נדל״ן",
        "params": {"סוגי_נכס": ["דירה"], "מס׳_חדרים": 3, "עיר": "ירושלים",
                   "מחיר": {"min": 0, "max": 1000000}},
        "confidence": 0.0,
        "notes": ["Stub response; query interpretation is not implemented."],
    }


@pytest.mark.parametrize("body", [{}, {"q": None}, {"q": 3}, {"q": []}, {"q": True},
                                  {"q": ""}, {"q": " \n\t "}, {"q": "א" * 2001},
                                  {"q": "דירה", "extra": 1}, [], None])
def test_invalid_requests(client, body):
    assert client.post("/parse", json=body).status_code == 422
    assert len(client.app.state.cache) == 0


def test_malformed_json(client):
    assert client.post("/parse", content="{", headers={"Content-Type": "application/json"}).status_code == 422


def test_health_and_openapi(client):
    assert client.get("/health").json() == {"status": "ok"}
    spec = client.get("/openapi.json").json()
    assert {"/parse", "/health", "/metrics"} <= spec["paths"].keys()
    assert client.get("/docs").status_code == 200


def test_cache_hit_metrics_and_whitespace():
    parser = CountingParser()
    with TestClient(create_app(parser=parser, cache_max_size=2)) as client:
        client.post("/parse", json={"q": " דירה "})
        client.post("/parse", json={"q": "דירה"})
        client.post("/parse", json={"q": ""})
        response = client.get("/metrics")
        text = response.text
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "parse_cache_hits_total 1.0" in text
        assert "parse_cache_misses_total 1.0" in text
        assert "parse_cache_entries 1.0" in text
        assert 'route="/parse",status="200"} 2.0' in text
        assert 'route="/parse",status="422"} 1.0' in text
        assert 'parse_requests_total{category="נדל״ן"} 2.0' in text
        assert 'le="0.15"' in text
        assert parser.calls == 1


def test_eviction_integration():
    parser = CountingParser()
    with TestClient(create_app(parser=parser, cache_max_size=2)) as client:
        for q in ("א", "ב", "א", "ג", "ב"):
            assert client.post("/parse", json={"q": q}).status_code == 200
        assert parser.calls == 4
        assert "parse_cache_evictions_total 2.0" in client.get("/metrics").text


def test_disabled_cache():
    parser = CountingParser()
    with TestClient(create_app(parser=parser, cache_max_size=0)) as client:
        for _ in range(2):
            client.post("/parse", json={"q": "דירה"})
        assert parser.calls == 2
        assert len(client.app.state.cache) == 0
        assert "parse_cache_hits_total 0.0" in client.get("/metrics").text


@pytest.mark.parametrize("value", ["bad", "-1", "1.5", ""])
def test_invalid_configuration(monkeypatch, value):
    monkeypatch.setenv("CACHE_MAX_SIZE", value)
    with pytest.raises(ValueError):
        create_app()


def test_environment_configuration(monkeypatch):
    monkeypatch.setenv("CACHE_MAX_SIZE", "7")
    assert create_app().state.cache.capacity == 7


def test_app_isolation():
    first, second = create_app(), create_app()
    assert first.state.metrics.registry is not second.state.metrics.registry
    first.state.cache.put("a", {"x": 1})
    assert len(second.state.cache) == 0


def test_cache_isolates_mutations():
    cache = LRUCache(2)
    value = {"nested": [1]}
    cache.put("a", value)
    value["nested"].append(2)
    cached = cache.get("a")
    cached["nested"].append(3)
    assert cache.get("a") == {"nested": [1]}


def test_cache_thread_safety():
    cache = LRUCache(4)

    def access(number):
        key = str(number % 8)
        cache.put(key, [number])
        result = cache.get(key)
        assert result is None or isinstance(result, list)

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(access, range(1000)))
    assert len(cache) <= 4


def test_concurrent_requests():
    async def run():
        app = create_app(cache_max_size=8)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(*[
                client.post("/parse", json={"q": f"דירה {i % 12}"}) for i in range(100)
            ])
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json()["confidence"] == 0 for r in responses)
        assert len(app.state.cache) <= 8

    asyncio.run(run())


def test_parser_failure_not_cached():
    class FailingParser:
        calls = 0

        async def parse(self, query):
            self.calls += 1
            raise RuntimeError(query)

    parser = FailingParser()
    with TestClient(create_app(parser=parser)) as client:
        for _ in range(2):
            response = client.post("/parse", json={"q": "private query"})
            assert response.status_code == 500
            assert response.json() == {"detail": "Internal server error"}
        assert parser.calls == 2
        assert len(client.app.state.cache) == 0
        assert 'route="/parse",status="500"} 2.0' in client.get("/metrics").text


def test_invalid_parser_output_not_cached():
    class InvalidParser:
        async def parse(self, query):
            return {"category": "unknown", "params": {}, "confidence": 5}

    with TestClient(create_app(parser=InvalidParser())) as client:
        assert client.post("/parse", json={"q": "דירה"}).status_code == 500
        assert len(client.app.state.cache) == 0


def test_logs_exclude_query(client, caplog):
    with caplog.at_level("INFO", logger="uvicorn.error"):
        client.post("/parse", json={"q": "private-query-123"})
    assert "private-query-123" not in caplog.text
    events = [json.loads(r.message) for r in caplog.records if r.name == "uvicorn.error"]
    assert events[-1]["cache"] == "miss"
    assert events[-1]["status"] == 200


def test_unmatched_paths_use_bounded_metric_label(client):
    client.get("/unknown-private-path")
    metrics = client.get("/metrics").text
    assert 'route="unmatched"' in metrics
    assert "unknown-private-path" not in metrics


@pytest.mark.parametrize("field,value", [("category", "unknown"), ("confidence", -0.1),
                                        ("confidence", 1.1), ("confidence", float("nan")),
                                        ("notes", [1])])
def test_response_contract(field, value):
    body = {"category": "רכב", "params": {}, "confidence": 0.0, "notes": []}
    body[field] = value
    with pytest.raises(ValidationError):
        ParseResponse.model_validate(body)


def test_stub_uses_taxonomy_fields(client):
    taxonomy = json.loads((Path(__file__).resolve().parents[2] / "yad2_search_taxonomy.json").read_text(encoding="utf-8"))
    result = client.post("/parse", json={"q": "דירה"}).json()
    category = taxonomy["קטגוריות"][result["category"]]
    assert set(result["params"]) <= set(category) | set(category["מאפיינים_כלליים"])
    assert result["params"]["סוגי_נכס"][0] in category["סוגי_נכס"]
