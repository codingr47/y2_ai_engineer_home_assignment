"""Compatibility ASGI entrypoint: python -m uvicorn src.app:app."""
from src.api.app import app, create_app

__all__ = ["app", "create_app"]
