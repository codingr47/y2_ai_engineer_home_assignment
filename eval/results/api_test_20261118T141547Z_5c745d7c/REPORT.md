# Parser evaluation

Mode: api; split: test; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 2181 |
| category_correct | 0.9935809261806511 |
| field_set_exact | 0.9371847776249427 |
| json_exact | 0.9005043558000917 |
| failures | {"http_400": 2, "http_502": 1} |
| latency_p50_seconds | 0.06869119999464601 |
| latency_p95_seconds | 3.1264642999740317 |
| successful_latency_p95_seconds | 3.1226149900525346 |
| tokens | {"input": 1611050.0, "output": 53049.0, "cached_input": 1610021.0, "reasoning": 35518.0} |

LLM fallback: 343 of 2181 requests (15.7%); routes {"local": 1836, "llm_fallback": 342, "llm_failed": 1, "rejected_no_fallback": 2}; source: X-OpenAI-Usage provider_called per request

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| רכב | 704 | 0.9915 | 0.9105 | 0.8764 |
| נדל״ן | 657 | 0.9970 | 0.9391 | 0.8782 |
| יד_שנייה | 720 | 0.9944 | 0.9556 | 0.9333 |
| None | 100 | 0.9800 | 0.9800 | 0.9800 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 751 | 0.9947 | 0.9481 | 0.9148 |
| slang | 717 | 0.9944 | 0.9428 | 0.9205 |
| typo | 710 | 0.9915 | 0.9211 | 0.8662 |
| typo+standard | 3 | 1.0000 | 0.6667 | 0.6667 |

## By route

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| local | 1836 | 0.9978 | 0.9619 | 0.9308 |
| llm_fallback | 342 | 0.9795 | 0.8129 | 0.7456 |
| rejected_no_fallback | 2 | 0.0000 | 0.0000 | 0.0000 |
| llm_failed | 1 | 0.0000 | 0.0000 | 0.0000 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "f83bb4a1e0f009427b0689869d0a45ed915cee1adb8d6a40496e812751915f24", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": 0.04272761, "status": "estimated", "reported_usage_only": true, "has_failed_requests": true, "billable_tokens": {"input": 1029, "output": 53049, "cached_input": 1610021, "cache_write_30m": 0, "cache_write_12h": 0}, "breakdown_usd": {"input": 0.0001029, "output": 0.0265245, "cached_input": 0.01610021, "cache_write_30m": 0.0, "cache_write_12h": 0.0}, "estimated_per_query_usd": 1.9590834479596514e-05, "projected_10m_queries_usd": 195.90834479596515, "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}`

Routing: `{"{\"destination\": \"local\", \"reason\": \"accepted\"}": 1697.0, "{\"destination\": \"fallback\", \"reason\": \"low_confidence\"}": 329.0, "{\"destination\": \"fallback\", \"reason\": \"empty_extraction\"}": 62.0, "{\"destination\": \"local\", \"reason\": \"non_search\"}": 93.0}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2181.0}`

Deployment: `{"image": "yad2-eval:e749df0c203f", "image_id": "sha256:a56ca5944c4639e46c55950abeaea5b5aba2b1578eb02f10893c4f49ab9e52e2", "container": "yad2-eval-e749df0c203f", "runtime": {"PARSER_MODE": "combined", "CACHE_MAX_SIZE": "0", "EXPOSE_TOKEN_USAGE": "true", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "0", "OPENAI_GUIDELINES_ENABLED": "true", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "LOCAL_PARSER": "tagger", "TAGGER_HYBRID_VEHICLE_TYPE": "false", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "low", "E5_MODEL_PATH": "/models/e5", "CLASSIFIER_ARTIFACT_PATH": "/app/src/artifacts/classifier.joblib", "CONFIDENCE_ARTIFACT_PATH": "/app/src/artifacts/confidence.joblib", "TAGGER_MODEL_DIR": "/models/tagger"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 1 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.
