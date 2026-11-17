# Parser evaluation

Mode: api; split: test; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 2181 |
| category_correct | 0.9958734525447043 |
| field_set_exact | 0.9160935350756534 |
| json_exact | 0.8794131132508024 |
| failures | {"http_400": 2} |
| latency_p50_seconds | 0.07700869999825954 |
| latency_p95_seconds | 3.578819799993653 |
| successful_latency_p95_seconds | 3.5826875399914533 |
| tokens | {"input": 2263314.0, "output": 63374.0, "cached_input": 2245638.0, "reasoning": 42516.0, "cache_write_30m": 16230.0, "cache_write_12h": 0.0} |

LLM fallback: 482 of 2181 requests (22.1%); routes {"local": 1697, "llm_fallback": 482, "rejected_no_fallback": 2}; source: X-OpenAI-Usage provider_called per request

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| רכב | 704 | 0.9943 | 0.8452 | 0.8139 |
| נדל״ן | 657 | 0.9985 | 0.9422 | 0.8782 |
| יד_שנייה | 720 | 0.9944 | 0.9500 | 0.9278 |
| None | 100 | 1.0000 | 1.0000 | 1.0000 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 751 | 0.9947 | 0.9108 | 0.8802 |
| slang | 717 | 0.9986 | 0.9289 | 0.9052 |
| typo | 710 | 0.9944 | 0.9099 | 0.8535 |
| typo+standard | 3 | 1.0000 | 0.6667 | 0.6667 |

## By route

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| local | 1697 | 0.9994 | 0.9364 | 0.9039 |
| llm_fallback | 482 | 0.9876 | 0.8485 | 0.7967 |
| rejected_no_fallback | 2 | 0.0000 | 0.0000 | 0.0000 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "f83bb4a1e0f009427b0689869d0a45ed915cee1adb8d6a40496e812751915f24", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": 0.05631673, "status": "estimated", "reported_usage_only": true, "has_failed_requests": true, "billable_tokens": {"input": 1446, "output": 63374, "cached_input": 2245638, "cache_write_30m": 16230, "cache_write_12h": 0}, "breakdown_usd": {"input": 0.0001446, "output": 0.031687, "cached_input": 0.02245638, "cache_write_30m": 0.00202875, "cache_write_12h": 0.0}, "estimated_per_query_usd": 2.5821517652453003e-05, "projected_10m_queries_usd": 258.21517652453, "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}`

Routing: `{"{\"destination\": \"local\", \"reason\": \"accepted\"}": 1697.0, "{\"destination\": \"fallback\", \"reason\": \"low_confidence\"}": 329.0, "{\"destination\": \"fallback\", \"reason\": \"empty_extraction\"}": 62.0, "{\"destination\": \"fallback\", \"reason\": \"unsupported_category\"}": 93.0}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2181.0}`

Deployment: `{"image": "yad2-eval:82d5d36d090c", "image_id": "sha256:0779b7eabf5e23e3df9c6be0bbf9465a84f1ecab7cc66d192bb5e4fa6c373d4f", "container": "yad2-eval-82d5d36d090c", "runtime": {"PARSER_MODE": "combined", "CACHE_MAX_SIZE": "0", "EXPOSE_TOKEN_USAGE": "true", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "0", "OPENAI_GUIDELINES_ENABLED": "true", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "LOCAL_PARSER": "tagger", "TAGGER_HYBRID_VEHICLE_TYPE": "true", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "low", "E5_MODEL_PATH": "/models/e5", "CLASSIFIER_ARTIFACT_PATH": "/app/src/artifacts/classifier.joblib", "CONFIDENCE_ARTIFACT_PATH": "/app/src/artifacts/confidence.joblib", "TAGGER_MODEL_DIR": "/models/tagger"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 1 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.
