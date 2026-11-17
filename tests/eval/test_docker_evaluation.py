import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest

from scripts import evaluate_docker as docker_eval
from scripts import evaluate_parser as evaluator
from src.config.settings import Settings


def test_docker_configuration_is_explicit():
    settings = Settings(openai_model="model", openai_reasoning_effort="high")
    config = docker_eval.runtime_configuration(settings, "openai")
    assert config["PARSER_MODE"] == "openai" and config["CACHE_MAX_SIZE"] == "0"
    assert config["OPENAI_MODEL"] == "model" and config["OPENAI_REASONING_EFFORT"] == "high"
    assert "OPENAI_API_KEY" not in config
    assert "OPENAI_REASONING_EFFORT" not in docker_eval.runtime_configuration(Settings(openai_model="model"), "openai")


def test_docker_run_builds_launches_evaluates_and_removes_only_owned_container(monkeypatch):
    calls = []

    async def docker(*args, **kwargs):
        calls.append((args, kwargs))
        if args[:2] == ("image", "inspect"):
            return "sha256:image"
        if args[0] == "port":
            return "127.0.0.1:12345"
        return "created"

    monkeypatch.setattr(docker_eval, "docker", docker)
    monkeypatch.setattr(docker_eval, "wait_ready", AsyncMock())
    evaluate = AsyncMock(return_value=("report", {}))
    monkeypatch.setattr(docker_eval, "run_api_evaluation", evaluate)
    settings = Settings(openai_model="luna", openai_api_key="private-key", openai_reasoning_effort="low")
    result = asyncio.run(docker_eval.run_docker_evaluation(settings=settings, allow_paid_calls=True))
    assert result == ("report", {})
    assert calls[0][0][0] == "build"
    run = next(call for call in calls if call[0][0] == "run")
    assert "private-key" not in " ".join(run[0])
    assert run[1]["env"]["OPENAI_API_KEY"] == "private-key"
    assert run[1]["env"]["PARSER_MODE"] == "openai"
    assert evaluate.call_args.kwargs["url"] == "http://127.0.0.1:12345"
    assert evaluate.call_args.kwargs["split"] == "test"
    assert "private-key" not in json.dumps(evaluate.call_args.kwargs["deployment"])
    assert calls[-1][0][:2] == ("rm", "--force")
    assert calls[-1][0][2] == run[0][run[0].index("--name") + 1]


def test_cleanup_after_readiness_failure(monkeypatch):
    command = AsyncMock(return_value="127.0.0.1:12345")
    monkeypatch.setattr(docker_eval, "docker", command)
    monkeypatch.setattr(docker_eval, "wait_ready", AsyncMock(side_effect=RuntimeError("not ready")))
    with pytest.raises(RuntimeError, match="not ready"):
        asyncio.run(docker_eval.run_docker_evaluation(mode="stub", settings=Settings()))
    assert command.call_args.args[:2] == ("rm", "--force")


def test_local_missing_encoder_fails_before_build(tmp_path, monkeypatch):
    command = AsyncMock()
    monkeypatch.setattr(docker_eval, "docker", command)
    with pytest.raises(ValueError, match="provisioned"):
        asyncio.run(docker_eval.run_docker_evaluation(mode="local", settings=Settings(e5_model_path=tmp_path / "missing")))
    command.assert_not_called()


def test_paid_optin_required_before_docker(monkeypatch):
    command = AsyncMock()
    monkeypatch.setattr(docker_eval, "docker", command)
    with pytest.raises(ValueError, match="paid-call"):
        asyncio.run(docker_eval.run_docker_evaluation())
    command.assert_not_called()


def test_http_evaluation_posts_dataset_rows_and_records_statuses(tmp_path, monkeypatch):
    rows = [dict(id="one", query="query one", domain=None, expected={"params": {}}, metadata={}),
            dict(id="two", query="query two", domain=None, expected={"params": {}}, metadata={})]
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/metrics":
            return httpx.Response(200, text="parse_cache_misses_total 0\n")
        assert request.method == "POST" and request.url.path == "/parse"
        if json.loads(request.content)["q"] == "query two":
            return httpx.Response(503, json={"detail": {"code": "provider_unavailable"}})
        return httpx.Response(200, json={"category": None, "params": {}, "confidence": .8, "notes": []})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(evaluator.httpx if hasattr(evaluator, "httpx") else httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    monkeypatch.setattr(evaluator, "load_rows", lambda _: rows)
    directory, summary = asyncio.run(evaluator.run_api_evaluation(
        url="http://evaluation", split="test", settings=Settings(), output=tmp_path,
        deployment={"image_id": "sha256:test"}))
    assert summary["overall"]["json_exact"] == .5
    assert summary["http_statuses"] == {"200": 1, "503": 1}
    assert summary["deployment"]["image_id"] == "sha256:test"
    assert [json.loads(request.content)["q"] for request in requests if request.url.path == "/parse"] == ["query one", "query two"]
    assert (directory / "REPORT.md").is_file()
    assert not list(tmp_path.glob("*.partial.jsonl"))
