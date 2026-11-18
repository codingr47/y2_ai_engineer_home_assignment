# Parser evaluation

Mode: api; split: test; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 2181 |
| category_correct | 0.9922054103622192 |
| field_set_exact | 0.9197615772581385 |
| json_exact | 0.8546538285190279 |
| failures | {"http_400": 2} |
| latency_p50_seconds | 0.06341269996482879 |
| latency_p95_seconds | 0.09883500006981194 |
| successful_latency_p95_seconds | 0.09887247006408867 |
| tokens | {"input": 98621.0, "output": 3278.0, "cached_input": 98558.0, "reasoning": 2083.0} |

LLM fallback: 21 of 2181 requests (1.0%); routes {"local": 2158, "llm_fallback": 21, "rejected_no_fallback": 2}; source: X-OpenAI-Usage provider_called per request

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| רכב | 704 | 0.9929 | 0.9162 | 0.8523 |
| נדל״ן | 657 | 0.9939 | 0.9102 | 0.8097 |
| יד_שנייה | 720 | 0.9986 | 0.9306 | 0.8875 |
| None | 100 | 0.9300 | 0.9300 | 0.9300 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 751 | 0.9947 | 0.9348 | 0.8855 |
| slang | 717 | 0.9888 | 0.9289 | 0.8759 |
| typo | 710 | 0.9930 | 0.8944 | 0.8014 |
| typo+standard | 3 | 1.0000 | 1.0000 | 0.6667 |

## By route

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| local | 2158 | 0.9935 | 0.9212 | 0.8577 |
| llm_fallback | 21 | 0.9524 | 0.8571 | 0.6190 |
| rejected_no_fallback | 2 | 0.0000 | 0.0000 | 0.0000 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "f83bb4a1e0f009427b0689869d0a45ed915cee1adb8d6a40496e812751915f24", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": 0.00263088, "status": "estimated", "reported_usage_only": true, "has_failed_requests": true, "billable_tokens": {"input": 63, "output": 3278, "cached_input": 98558, "cache_write_30m": 0, "cache_write_12h": 0}, "breakdown_usd": {"input": 6.3e-06, "output": 0.001639, "cached_input": 0.00098558, "cache_write_30m": 0.0, "cache_write_12h": 0.0}, "estimated_per_query_usd": 1.2062723521320495e-06, "projected_10m_queries_usd": 12.062723521320494, "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}`

Routing: `{"{\"destination\": \"local\", \"reason\": \"accepted\"}": 2026.0, "{\"destination\": \"fallback\", \"reason\": \"empty_extraction\"}": 62.0, "{\"destination\": \"local\", \"reason\": \"non_search\"}": 93.0}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2181.0}`

Deployment: `{"image": "yad2-eval:44bbb4a44f96", "image_id": "sha256:129dcbcd82c528accede096651af593a07d456b08105cd23bf0f14d1e3c00abc", "container": "yad2-eval-44bbb4a44f96", "runtime": {"PARSER_MODE": "combined", "CACHE_MAX_SIZE": "0", "EXPOSE_TOKEN_USAGE": "true", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "0", "OPENAI_GUIDELINES_ENABLED": "true", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "LOCAL_PARSER": "tagger", "TAGGER_HYBRID_VEHICLE_TYPE": "false", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "low", "PARSE_CONFIDENCE_THRESHOLD": "0.0", "E5_MODEL_PATH": "/models/e5", "CLASSIFIER_ARTIFACT_PATH": "/app/src/artifacts/classifier.joblib", "CONFIDENCE_ARTIFACT_PATH": "/app/src/artifacts/confidence.joblib", "TAGGER_MODEL_DIR": "/models/tagger"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 1 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.
