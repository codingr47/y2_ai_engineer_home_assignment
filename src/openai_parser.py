"""Async OpenAI fallback with taxonomy validation and sanitized failures."""
import json
from time import perf_counter

from src.openai_catalog import Catalog
from src.few_shots import prompt_configuration
from src.request_usage import provider_started, record_tokens
from src.injection_guard import check_query, PromptInjectionDetected

UNCALIBRATED_NOTE = "OpenAI fallback; confidence is a model self-rating and is uncalibrated."


class ProviderFailure(Exception):
    def __init__(self, code, status_code):
        self.code = code
        self.status_code = status_code
        super().__init__(code)


def create_client(settings):
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value(),
                       timeout=settings.openai_timeout_seconds, max_retries=0)


class OpenAIParser:
    def __init__(self, client, settings, metrics=None, catalog=None):
        self.client, self.settings, self.metrics = client, settings, metrics
        self.catalog = catalog or Catalog()
        self.system_prompt, self.prompt_metadata = prompt_configuration(
            self.catalog, settings.openai_few_shot_count, guidelines_enabled=settings.openai_guidelines_enabled)
        self.last_usage = {}

    async def parse(self, query):
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        started, outcome = perf_counter(), "error"
        self.last_usage = {}
        try:
            check_query(query)
        except PromptInjectionDetected as exc:
            if self.metrics:
                self.metrics.security_rejections.labels(exc.reason).inc()
            raise
        try:
            kwargs = dict(model=self.settings.openai_model, instructions=self.system_prompt,
                          input=[{"role": "user", "content": "Return JSON for QUERY:\n" + json.dumps(query, ensure_ascii=False)}],
                          text={"format": {"type": "json_object"}}, store=False,
                          max_output_tokens=self.settings.openai_max_output_tokens)
            if self.settings.openai_reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": self.settings.openai_reasoning_effort}
            provider_started()
            try:
                response = await self.client.responses.create(**kwargs)
            except APITimeoutError:
                raise ProviderFailure("provider_timeout", 503) from None
            except APIConnectionError:
                raise ProviderFailure("provider_unavailable", 503) from None
            except APIStatusError as exc:
                code = "provider_unavailable" if exc.status_code == 429 or exc.status_code >= 500 else "provider_configuration_error"
                raise ProviderFailure(code, 503) from None
            usage = getattr(response, "usage", None)
            if usage:
                details = getattr(usage, "input_tokens_details", None)
                output_details = getattr(usage, "output_tokens_details", None)
                self.last_usage = {"input": usage.input_tokens, "output": usage.output_tokens,
                                   "cached_input": getattr(details, "cached_tokens", 0) or 0,
                                   "reasoning": getattr(output_details, "reasoning_tokens", 0) or 0}
                write_total = getattr(details, "cache_write_tokens", 0) or 0
                write_details = getattr(details, "cache_write_tokens_details", None)
                if write_total:
                    write_12h = (write_details.get("ttl_12h_tokens", 0) if isinstance(write_details, dict)
                                 else getattr(write_details, "ttl_12h_tokens", 0)) or 0
                    self.last_usage.update(cache_write_30m=write_total - write_12h, cache_write_12h=write_12h)
                record_tokens(self.last_usage)
                if self.metrics:
                    for kind, count in self.last_usage.items():
                        self.metrics.provider_tokens.labels(kind).inc(count)
            if any(getattr(part, "type", None) == "refusal" for item in response.output
                   for part in getattr(item, "content", [])):
                raise ProviderFailure("provider_refusal", 502)
            if response.status != "completed":
                raise ProviderFailure("provider_incomplete", 502)
            try:
                def unique_object(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ValueError("duplicate JSON key")
                        result[key] = value
                    return result

                payload = json.loads(response.output_text, object_pairs_hook=unique_object)
                if not isinstance(payload, dict) or set(payload) != {"category", "params", "confidence"}:
                    raise ValueError("invalid envelope")
                result = self.catalog.validate(payload)
            except (ValueError, TypeError, KeyError):
                raise ProviderFailure("provider_invalid_output", 502) from None
            result.notes = [UNCALIBRATED_NOTE]
            outcome = "success"
            return result
        except ProviderFailure as exc:
            outcome = exc.code
            raise
        finally:
            if self.metrics:
                self.metrics.provider_calls.labels(outcome).inc()
                self.metrics.provider_duration.observe(perf_counter() - started)
