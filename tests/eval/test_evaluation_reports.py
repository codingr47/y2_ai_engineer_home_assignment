"""Offline coverage of evaluation scoring, failures, and report artifacts."""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from scripts import evaluate_parser as evaluation
from src.http_models import SearchResponse
from src.openai_parser import ProviderFailure
from src.settings import Settings


def rows():
    return [dict(id=str(i), query="שאילתה", domain=domain,
                 expected={"params": params}, metadata={"language": ["standard"]})
            for i, (domain, params) in enumerate([(None, {}), ("רכב", {"יצרנים": "טויוטה"}),
                                                 ("נדל״ן", {"מחיר": {"max": 100}})])]


def test_scoring_counts_failures_and_nonsearch_strictly(tmp_path):
    parser = NS(parse=AsyncMock(side_effect=[
        SearchResponse(category="רכב", params={}, confidence=.8),
        SearchResponse(category="רכב", params={"יצרן": "טויוטה"}, confidence=.9),
        ProviderFailure("provider_timeout", 503)]))
    records = asyncio.run(evaluation.evaluate(parser, rows()))
    assert [r["json_exact"] for r in records] == [False, True, False]
    summary = evaluation.summarize(records)
    assert summary["overall"]["json_exact"] == 1 / 3
    assert summary["overall"]["failures"] == {"provider_timeout": 1}
    summary.update(configuration=dict(mode="openai", split="val", model="one", reasoning_effort="low"), routing={})
    directory = tmp_path / "report"
    evaluation.write_reports(directory, summary, records)
    assert json.loads((directory / "summary.json").read_text(encoding="utf-8"))["overall"]["rows"] == 3
    assert len((directory / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert "By category" in (directory / "REPORT.md").read_text(encoding="utf-8")
    assert "Query results" in (directory / "REPORT.html").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluation.write_reports(directory, summary, records)


def test_null_nonsearch_is_correct():
    parser = NS(parse=AsyncMock(return_value=SearchResponse(category=None, params={}, confidence=.8)))
    records = asyncio.run(evaluation.evaluate(parser, rows()[:1]))
    assert records[0]["json_exact"] and records[0]["category_correct"]


def test_paid_optin_required():
    with pytest.raises(ValueError, match="paid-call"):
        asyncio.run(evaluation.run_evaluation(mode="openai"))


def test_full_report_with_mocked_provider_and_failed_row(tmp_path, monkeypatch):
    from src.openai_catalog import Catalog
    monkeypatch.setattr(evaluation, "load_rows", lambda _: rows())
    client = NS(close=AsyncMock())
    monkeypatch.setattr(evaluation, "create_client", lambda _: client)

    class FakeProvider:
        def __init__(self, client, settings, metrics):
            self.catalog = Catalog()
            self.system_prompt = "fixed prompt"
            self.last_usage = {}
            self.parse = AsyncMock(side_effect=[SearchResponse(category=None, params={}, confidence=.8),
                                               ProviderFailure("provider_timeout", 503),
                                               SearchResponse(category="נדל״ן", params={"מחיר": {"max": 100}}, confidence=.9)])

    monkeypatch.setattr(evaluation, "OpenAIParser", FakeProvider)
    directory, summary = asyncio.run(evaluation.run_evaluation(
        mode="openai", settings=Settings(openai_api_key="secret", openai_model="model-two", openai_reasoning_effort="high"),
        output=tmp_path, allow_paid_calls=True))
    assert summary["overall"]["json_exact"] == 2 / 3
    assert summary["configuration"]["reasoning_effort"] == "high"
    assert summary["configuration"]["model"] == "model-two"
    assert len(summary["configuration"]["dataset_sha256"]) == 64
    assert "secret" not in (directory / "summary.json").read_text(encoding="utf-8")
    client.close.assert_awaited_once()


def test_evaluation_validation_rows_available():
    loaded = evaluation.load_rows("val")
    assert loaded and all("domain" in row and "params" in row["expected"] for row in loaded)


def test_html_escapes_dataset_and_model_content():
    from scripts.evaluation_html import render_html
    record = {"id": "one", "query": "<script>alert('query')</script>", "gold": {}, "pred": {},
              "json_exact": False, "error": None, "latency_seconds": .2}
    summary = {"configuration": {"mode": "api", "split": "test", "model": '<img src=x onerror="alert(1)">',
                                  "reasoning_effort": "low"},
               "overall": {"rows": 1, "json_exact": 0, "failures": {}}, "slices": {}}
    html = render_html(summary, [record])
    assert "<script>alert(" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html
    assert 'id="search"' in html and 'data-outcome="incorrect"' in html
