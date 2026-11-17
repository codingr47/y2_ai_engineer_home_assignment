import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from scripts import evaluate_few_shots as comparison
from scripts.extraction_rca import analyze, render
from scripts.evaluate_docker import runtime_configuration
from src.few_shots import EXAMPLES, prompt_configuration
from src.openai_catalog import Catalog
from src.openai_parser import OpenAIParser
from src.openai_prompt import EXTRACTOR_SYSTEM_TEMPLATE
from src.openai_prompt import IMPORTANT_GUIDELINES
from src.settings import Settings


def test_zero_shot_prompt_is_byte_identical_to_baseline():
    catalog = Catalog()
    prompt, metadata = prompt_configuration(catalog, 0)
    assert prompt == EXTRACTOR_SYSTEM_TEMPLATE.format(catalog=catalog.prompt())
    assert metadata["prompt_sha256"] == "efe218dabaa4213940ac9ec3bcd6aeb7e4f9fd8c56d7b01f6272aa5144b02b67"
    assert metadata["example_ids"] == []


def test_guidelines_are_exactly_inserted_without_changing_catalog_or_examples():
    catalog = Catalog()
    old, previous = prompt_configuration(catalog, 12, guidelines_enabled=False)
    new, metadata = prompt_configuration(catalog, 12, guidelines_enabled=True)
    assert new.replace(IMPORTANT_GUIDELINES + "\n", "", 1) == old
    assert new.split("CATALOG\n", 1)[1] == catalog.prompt()
    assert metadata["guidelines_enabled"] is True
    assert metadata["example_ids"] == previous["example_ids"]
    assert metadata["prompt_sha256"] != previous["prompt_sha256"]


def test_guidelines_default_on_and_environment_toggle_forwarded(monkeypatch):
    assert Settings().openai_guidelines_enabled
    monkeypatch.setenv("OPENAI_GUIDELINES_ENABLED", "false")
    settings = Settings.from_env()
    assert not settings.openai_guidelines_enabled
    assert runtime_configuration(settings, "openai")["OPENAI_GUIDELINES_ENABLED"] == "false"


@pytest.mark.parametrize("count", [0, 6, 12])
def test_demonstration_counts_catalog_integrity_and_stable_hash(count):
    catalog = Catalog()
    prompt, metadata = prompt_configuration(catalog, count)
    assert len(metadata["example_ids"]) == count
    assert prompt.split("CATALOG\n", 1)[1] == catalog.prompt()
    assert metadata["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert prompt == prompt_configuration(catalog, count)[0]
    assert metadata["example_ids"] == [e["source_id"] for e in json.loads(EXAMPLES.read_text(encoding="utf-8"))["examples"][:count]]


def test_examples_preserve_reviewed_training_provenance_and_no_heldout_overlap():
    result = comparison.validate_examples()
    assert result["examples"] == 12
    assert len({c["prompt_sha256"] for c in result["configurations"].values()}) == 3


@pytest.mark.parametrize("count", [-1, 1, 5, 7, 13, True, "bad"])
def test_bad_counts_rejected(count):
    with pytest.raises((ValueError, ValidationError)):
        Settings(openai_few_shot_count=count)


def test_environment_count_and_docker_forwarding(monkeypatch):
    monkeypatch.setenv("OPENAI_FEW_SHOT_COUNT", "6")
    settings = Settings.from_env()
    assert settings.openai_few_shot_count == 6
    assert runtime_configuration(settings, "openai")["OPENAI_FEW_SHOT_COUNT"] == "6"


def test_examples_are_in_instructions_and_not_user_input():
    client = NS(responses=NS(create=AsyncMock(return_value=NS(
        usage=None, status="completed", output=[], output_text='{"category":null,"params":{},"confidence":0.8}'))))
    parser = OpenAIParser(client, Settings(openai_few_shot_count=12))
    asyncio.run(parser.parse("private-user-input"))
    request = client.responses.create.call_args.kwargs
    assert "EXPECTED EXTRACTION" in request["instructions"]
    assert "EXPECTED EXTRACTION" not in request["input"][0]["content"]
    assert "private-user-input" not in request["instructions"]
    assert "\"confidence\"" not in request["instructions"].split("REVIEWED EXTRACTION EXAMPLES", 1)[1]


def record(identifier, expected, prediction, error=None):
    return {"id": identifier, "query": "query <script>alert(1)</script>", "gold": expected, "pred": prediction,
            "error": error, "json_exact": False}


def test_rca_date_equivalence_and_unknown_rejected_output_are_honest():
    rows = [record("date", {"category": "נדל״ן", "params": {"תאריך_כניסה": "01-12-2026"}},
                           {"category": "נדל״ן", "params": {"תאריך_כניסה": "2026-12-01"}}),
            record("failed", {"category": "רכב", "params": {}}, None, "http_502")]
    before = copy.deepcopy(rows)
    result = analyze(rows)
    assert result["failure_rows"] == 2
    assert result["failures"][0]["review_status"] == "format_equivalence_confirmed"
    assert result["failures"][1]["review_status"] == "unresolved_provider_output"
    assert not any(r["demonstration_eligible"] for r in result["failures"])
    assert rows == before
    assert "<script>alert(" not in render(result)


def test_paired_changes_require_identical_data_and_report_both_directions():
    baseline = {str(i): {"gold": {}, "query": str(i), "json_exact": i == 0} for i in range(2)}
    candidate = {str(i): {"gold": {}, "query": str(i), "json_exact": i == 1} for i in range(2)}
    assert comparison.paired_changes(baseline, candidate) == {"corrected_ids": ["1"], "newly_broken_ids": ["0"]}
    candidate["0"]["query"] = "different"
    with pytest.raises(ValueError, match="identical"):
        comparison.paired_changes(baseline, candidate)


def test_selection_uses_accuracy_then_fewer_examples():
    runs = [{"few_shot_count": count, "summary": {"overall": {"json_exact": exact, "latency_p95_seconds": latency}}}
            for count, exact, latency in [(0, .8, 1), (6, .9, 4), (12, .9, 2)]]
    assert comparison.select_winner(runs)["few_shot_count"] == 6


def test_comparison_report_resolves_relative_directory(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    directory = Path("comparison")
    directory.mkdir()
    report = {"settings": {}, "selection_rule": "accuracy", "provenance": {},
              "validation_runs": [{"few_shot_count": 0, "directory": str((directory / "run").resolve()),
                                    "summary": {"overall": {"json_exact": .9, "latency_p95_seconds": 1, "failures": {}},
                                                "cost": {"estimated_total_usd": .1}}}]}
    comparison.write_comparison(directory, report)
    assert 'href="run/REPORT.html"' in (directory / "REPORT.html").read_text(encoding="utf-8")
