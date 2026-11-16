"""Async parser boundary shared by local parsing, injected fallback, and the default stub."""

from typing import Protocol

from src.models import FallbackDetail, ParseResponse, StubParams


class Parser(Protocol):
    async def parse(self, query: str) -> ParseResponse: ...


class FallbackRequired(Exception):
    """A valid request needs a fallback parser that is not configured."""

    def __init__(self, reason, confidence=None, threshold=None):
        self.detail = FallbackDetail(reason=reason, confidence=confidence, threshold=threshold).model_dump()
        super().__init__(reason)


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
