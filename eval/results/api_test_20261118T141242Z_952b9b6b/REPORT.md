# Parser evaluation

Mode: api; split: test; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 2181 |
| category_correct | 0.9830353049060064 |
| field_set_exact | 0.911508482347547 |
| json_exact | 0.8486932599724897 |
| failures | {"http_503": 23} |
| latency_p50_seconds | 0.06422609998844564 |
| latency_p95_seconds | 0.09872929996345192 |
| successful_latency_p95_seconds | 0.0987366949848365 |
| tokens | {} |

LLM fallback: 0 of 2181 requests (0.0%); routes {"local": 2158, "rejected_no_fallback": 23}; source: X-OpenAI-Usage provider_called per request

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| רכב | 704 | 0.9915 | 0.9148 | 0.8509 |
| נדל״ן | 657 | 0.9909 | 0.9072 | 0.8097 |
| יד_שנייה | 720 | 0.9750 | 0.9097 | 0.8708 |
| None | 100 | 0.9300 | 0.9300 | 0.9300 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 751 | 0.9854 | 0.9268 | 0.8802 |
| slang | 717 | 0.9847 | 0.9247 | 0.8731 |
| typo | 710 | 0.9789 | 0.8817 | 0.7915 |
| typo+standard | 3 | 1.0000 | 1.0000 | 0.6667 |

## By route

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| local | 2158 | 0.9935 | 0.9212 | 0.8577 |
| rejected_no_fallback | 23 | 0.0000 | 0.0000 | 0.0000 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "f83bb4a1e0f009427b0689869d0a45ed915cee1adb8d6a40496e812751915f24", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": null, "status": "unavailable", "reason": "token_usage_not_reported"}`

Routing: `{"{\"destination\": \"local\", \"reason\": \"accepted\"}": 2026.0, "{\"destination\": \"fallback\", \"reason\": \"empty_extraction\"}": 62.0, "{\"destination\": \"local\", \"reason\": \"non_search\"}": 93.0}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2181.0}`

Deployment: `{"image": "yad2-eval:2a23b940f07e", "image_id": "sha256:69917d672ca1d1852865280eb9e8888ddf1c8c1990ff49692a8923f58ee1f67f", "container": "yad2-eval-2a23b940f07e", "runtime": {"PARSER_MODE": "local", "CACHE_MAX_SIZE": "0", "EXPOSE_TOKEN_USAGE": "true", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "0", "OPENAI_GUIDELINES_ENABLED": "true", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "LOCAL_PARSER": "tagger", "TAGGER_HYBRID_VEHICLE_TYPE": "false", "PARSE_CONFIDENCE_THRESHOLD": "0.0", "E5_MODEL_PATH": "/models/e5", "CLASSIFIER_ARTIFACT_PATH": "/app/src/artifacts/classifier.joblib", "CONFIDENCE_ARTIFACT_PATH": "/app/src/artifacts/confidence.joblib", "TAGGER_MODEL_DIR": "/models/tagger"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 1 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.
