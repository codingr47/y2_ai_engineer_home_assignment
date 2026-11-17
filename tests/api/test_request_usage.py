import asyncio
import json
from types import SimpleNamespace as NS

from fastapi.testclient import TestClient
import httpx

from src.api.app import create_app
from src.llm.parser import OpenAIParser
from src.config.settings import Settings


def fake_response(count, status="completed"):
    return NS(status=status, output=[], output_text='{"category":null,"params":{},"confidence":0.8}',
              usage=NS(input_tokens=count, output_tokens=count // 10,
                       input_tokens_details=NS(cached_tokens=0), output_tokens_details=NS(reasoning_tokens=1)))


def test_concurrent_usage_is_attributed_to_each_request():
    async def send_to_provider(**kwargs):
        query = json.loads(kwargs["input"][0]["content"].split("\n", 1)[1])
        await asyncio.sleep(.02 if query == "left" else 0)
        return fake_response(111 if query == "left" else 222)

    settings = Settings(expose_token_usage=True)
    parser = OpenAIParser(NS(responses=NS(create=send_to_provider)), settings)
    app = create_app(parser=parser, settings=settings, cache_max_size=0)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            responses = await asyncio.gather(client.post("/parse", json={"q": "left"}), client.post("/parse", json={"q": "right"}))
        assert [json.loads(r.headers["X-OpenAI-Usage"])["tokens"]["input"] for r in responses] == [111, 222]
        assert all(json.loads(r.headers["X-OpenAI-Usage"])["provider_called"] for r in responses)

    asyncio.run(exercise())


def test_cache_hit_does_not_reuse_billed_token_usage():
    async def send_to_provider(**kwargs):
        return fake_response(100)

    settings = Settings(expose_token_usage=True)
    parser = OpenAIParser(NS(responses=NS(create=send_to_provider)), settings)
    with TestClient(create_app(parser=parser, settings=settings)) as client:
        first = client.post("/parse", json={"q": "same query"})
        second = client.post("/parse", json={"q": "same query"})
    assert json.loads(first.headers["X-OpenAI-Usage"])["tokens"]["input"] == 100
    assert json.loads(second.headers["X-OpenAI-Usage"]) == {"provider_called": False, "tokens": None}


def test_failed_parse_still_exposes_reported_billed_usage_when_enabled():
    async def send_to_provider(**kwargs):
        return fake_response(100, "incomplete")

    settings = Settings(expose_token_usage=True)
    parser = OpenAIParser(NS(responses=NS(create=send_to_provider)), settings)
    with TestClient(create_app(parser=parser, settings=settings)) as client:
        response = client.post("/parse", json={"q": "query"})
    assert response.status_code == 502
    assert json.loads(response.headers["X-OpenAI-Usage"])["tokens"]["input"] == 100


def test_usage_export_is_off_by_default():
    with TestClient(create_app(settings=Settings())) as client:
        assert "X-OpenAI-Usage" not in client.post("/parse", json={"q": "query"}).headers
