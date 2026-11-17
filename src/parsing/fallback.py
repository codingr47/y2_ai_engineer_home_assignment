"""Escalation without altering calibrated candidate generation or gate bindings."""
from src.parsing.parser import FallbackRequired


class FallbackRouter:
    def __init__(self, local, fallback):
        self.local = local
        self.fallback = fallback

    async def parse(self, query):
        try:
            return await self.local.parse(query)
        except FallbackRequired:
            return await self.fallback.parse(query)
