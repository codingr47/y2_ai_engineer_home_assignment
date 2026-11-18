"""ASGI application. Run with: python -m uvicorn src.api.app:app."""

import json
import logging
import os
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.infrastructure.cache import LRUCache
from src.infrastructure.metrics import Metrics
from src.contracts.models import FallbackResponse, ParseRequest, ParseResponse
from src.contracts.http_models import ProviderErrorResponse, SearchResponse
from src.llm.parser import OpenAIParser, ProviderFailure, create_client
from src.parsing.fallback import FallbackRouter
from src.parsing.validated import CatalogCheckedParser
from src.parsing.parser import FallbackRequired, Parser, StubParser
from src.config.settings import Settings
from src.infrastructure import request_usage
from src.llm.injection_guard import PromptInjectionDetected

logger = logging.getLogger("uvicorn.error")


def create_app(*, cache_max_size: int | None = None, parser: Parser | None = None,
               fallback_parser: Parser | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings.from_env()
    if cache_max_size is None:
        try:
            cache_max_size = int(os.getenv("CACHE_MAX_SIZE", "1024"))
        except ValueError as exc:
            raise ValueError("CACHE_MAX_SIZE must be a nonnegative integer") from exc
    cache = LRUCache[SearchResponse](cache_max_size)
    metrics = Metrics()
    mode = settings.active_parser_mode

    @asynccontextmanager
    async def lifespan(app):
        client = None
        try:
            if parser is None and mode == "openai":
                client = create_client(settings)
                app.state.parser = OpenAIParser(client, settings, metrics)
            elif parser is None and mode in {"local", "combined"}:
                import anyio
                if settings.local_parser == "tagger":      # the fine-tuned tagger (docs/tagger-parser.md)
                    from src.tagger.parser import TaggerParser as Local
                else:                                        # Frozen Fusion
                    from src.frozen_fusion.local_parser import LocalParser as Local
                local = await anyio.to_thread.run_sync(lambda: Local.load(settings, fallback_parser, metrics))
                if fallback_parser is None:   # an injected fallback receives empty answers directly, as before
                    from src.parsing.empty_params import EmptyParamsParser
                    local = EmptyParamsParser(local, metrics)
                # Same closed-catalog check as the LLM path; invalid local answers escalate (src/parsing/validated.py).
                app.state.parser = CatalogCheckedParser(local, metrics=metrics, fallback=fallback_parser)
                if fallback_parser is None and mode == "combined":
                    client = create_client(settings)
                    app.state.parser = FallbackRouter(app.state.parser, OpenAIParser(client, settings, metrics))
            yield
        finally:
            if client is not None:
                await client.close()

    app = FastAPI(title="Yad2 Search API", version="0.2.0",
                  description="Optional local search parsing with confidence routing", lifespan=lifespan)
    app.state.cache = cache
    app.state.metrics = metrics
    app.state.parser = parser if parser is not None else (StubParser() if mode == "stub" else None)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = perf_counter()
        usage, usage_token = request_usage.begin()
        try:
            response = await call_next(request)
        except Exception:
            # Avoid logging exception messages, which could contain query text.
            response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        finally:
            request_usage.reset(usage_token)
        if settings.expose_token_usage:
            response.headers["X-OpenAI-Usage"] = json.dumps(usage, separators=(",", ":"))
        elapsed = perf_counter() - started
        route = request.scope.get("route")
        route_label = route.path if route is not None else "unmatched"
        method = request.method if request.method in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"} else "OTHER"
        metrics.requests.labels(route_label, method, str(response.status_code)).inc()
        metrics.duration.labels(route_label, method).observe(elapsed)
        logger.info(json.dumps({
            "event": "http_request", "route": route_label, "method": method,
            "status": response.status_code, "elapsed_seconds": elapsed,
            "category": getattr(request.state, "category", None),
            "cache": getattr(request.state, "cache_outcome", "not_applicable"),
        }, ensure_ascii=False))
        return response

    @app.post("/parse", response_model=SearchResponse,
              responses={400: {"model": ProviderErrorResponse,
                               "description": "LLM input guard detected a prompt injection attempt"},
                         503: {"model": FallbackResponse | ProviderErrorResponse,
                               "description": "Fallback unavailable or provider configuration rejected"},
                         502: {"model": ProviderErrorResponse,
                               "description": "Invalid, incomplete, or refused provider response"}})
    async def parse(body: ParseRequest, request: Request) -> SearchResponse | JSONResponse:
        result = cache.get(body.q)
        if result is not None:
            metrics.hits.inc()
            request.state.cache_outcome = "hit"
        else:
            metrics.misses.inc()
            request.state.cache_outcome = "miss" if cache.capacity else "disabled"
            try:
                parsed = await app.state.parser.parse(body.q)
                result = SearchResponse.model_validate(parsed.model_dump() if isinstance(parsed, ParseResponse) else parsed)
            except PromptInjectionDetected as exc:
                logger.info(json.dumps({"event": "llm_security_rejection", "reason": exc.reason}))
                return JSONResponse(status_code=400, content={"detail": {"code": "prompt_injection_detected"}})
            except FallbackRequired as exc:
                logger.info(json.dumps({"event": "parse_decision", **exc.detail}))
                return JSONResponse(status_code=503, content={"detail": exc.detail})
            except ProviderFailure as exc:
                logger.info(json.dumps({"event": "provider_failure", "code": exc.code}))
                return JSONResponse(status_code=exc.status_code, content={"detail": {"code": exc.code}})
            if cache.put(body.q, result):
                metrics.evictions.inc()
            metrics.entries.set(len(cache))
        request.state.category = result.category
        metrics.parses.labels(result.category or "none").inc()
        return result

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=Response)
    async def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return app


app = create_app()
