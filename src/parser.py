"""Replaceable parser boundary; deliberately no query interpretation yet."""

from typing import Protocol

from src.models import ParseResponse, StubParams


class Parser(Protocol):
    async def parse(self, query: str) -> ParseResponse: ...


class StubParser:
    async def parse(self, query: str) -> ParseResponse:
        params = StubParams.model_validate({
            "סוגי_נכס": ["דירה"], "מס׳_חדרים": 3,
            "עיר": "ירושלים", "מחיר": {"min": 0, "max": 1000000},
        })
        return ParseResponse(
            category="נדל״ן", params=params.model_dump(by_alias=True), confidence=0.0,
            notes=["Stub response; query interpretation is not implemented."],
        )
