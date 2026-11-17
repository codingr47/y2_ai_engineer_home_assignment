"""Local answers get the same deterministic taxonomy check as LLM answers; invalid ones escalate."""
import json
from pathlib import Path

import anyio
import pytest

from src.contracts.models import ParseResponse
from src.contracts.http_models import SearchResponse
from src.infrastructure.metrics import Metrics
from src.llm.catalog import Catalog
from src.parsing.fallback import FallbackRouter
from src.parsing.parser import FallbackRequired
from src.parsing.validated import CatalogCheckedParser, catalog_violation

CATALOG = Catalog()
VALID = ParseResponse(category="רכב", params={"יצרן": "טויוטה", "דגם": "קורולה", "שנה": {"min": 2020}}, confidence=0.9)
# The three local answers of the recorded test run that break the taxonomy (all three were wrong answers).
INVALID = {
    "invalid enum": ParseResponse(category="יד_שנייה", confidence=0.9, params={
        "סקטורים": "ריהוט", "תתי_קטגוריות": "סלון", "סוגי_רהיט": "שולחן", "חומר": "זכוכית", "עיר": "עפולה"}),
    "unknown or inactive field": ParseResponse(category="יד_שנייה", confidence=0.9, params={
        "סקטורים": "ריהוט", "תתי_קטגוריות": "סלון", "רזולוציה": "4K", "אזור": "שרון"}),
    "hierarchy must be scalar": ParseResponse(category="רכב", confidence=0.9, params={
        "יצרן": "מרצדס", "דגם": ["A-Class", "קונה"]}),
}
LLM_ANSWER = ParseResponse(category="רכב", params={"יצרן": "מרצדס"}, confidence=0.5, notes=["llm"])


class Fixed:
    def __init__(self, response):
        self.response, self.calls = response, 0

    async def parse(self, query):
        self.calls += 1
        return self.response


def run(parser):
    return anyio.run(parser.parse, "q")


def test_valid_and_non_search_answers_pass_unchanged():
    assert catalog_violation(CATALOG, VALID) is None
    assert catalog_violation(CATALOG, SearchResponse(category=None, params={}, confidence=0.9)) is None
    assert run(CatalogCheckedParser(Fixed(VALID), CATALOG)) is VALID


@pytest.mark.parametrize("reason", sorted(INVALID))
def test_invalid_local_answers_are_detected(reason):
    assert catalog_violation(CATALOG, INVALID[reason]) == reason


@pytest.mark.parametrize("reason", sorted(INVALID))
def test_router_escalates_invalid_local_answers(reason):
    metrics, fallback = Metrics(), Fixed(LLM_ANSWER)
    router = FallbackRouter(CatalogCheckedParser(Fixed(INVALID[reason]), CATALOG, metrics), fallback)
    assert run(router) is LLM_ANSWER and fallback.calls == 1
    assert metrics.registry.get_sample_value("parse_output_rejections_total", {"source": "local"}) == 1


def test_local_only_mode_refuses_instead_of_returning_invalid_output():
    with pytest.raises(FallbackRequired) as caught:
        run(CatalogCheckedParser(Fixed(INVALID["invalid enum"]), CATALOG))
    assert caught.value.detail["reason"] == "invalid_output"


def test_injected_fallback_is_used_directly():
    fallback = Fixed(LLM_ANSWER)
    assert run(CatalogCheckedParser(Fixed(INVALID["invalid enum"]), CATALOG, fallback=fallback)) is LLM_ANSWER


def test_every_correct_recorded_local_answer_passes():
    """No false rejections: every exactly-correct local answer of the recorded runs passes the check."""
    runs = sorted(Path("eval/results").glob("api_test_*/records.jsonl"))
    if not runs:
        pytest.skip("no recorded evaluation runs")
    checked = 0
    for path in runs:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("route") == "local" and row.get("json_exact"):
                pred = row["pred"]
                response = SearchResponse(category=pred["category"], params=pred["params"], confidence=0.5)
                assert catalog_violation(CATALOG, response) is None, row["id"]
                checked += 1
    assert checked > 0
