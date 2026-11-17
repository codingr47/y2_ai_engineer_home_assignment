# Parser evaluation

Mode: api; split: test; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 2181 |
| category_correct | 0.9926639156350298 |
| field_set_exact | 0.9275561668959194 |
| json_exact | 0.889958734525447 |
| failures | {"http_400": 2, "http_502": 1} |
| latency_p50_seconds | 0.07418709999183193 |
| latency_p95_seconds | 2.9983181999996305 |
| successful_latency_p95_seconds | 2.998606665004627 |
| tokens | {"input": 1827072.0, "output": 59514.0, "cached_input": 1807692.0, "reasoning": 40761.0, "cache_write_30m": 18213.0, "cache_write_12h": 0.0} |

LLM fallback: 389 of 2181 requests (17.8%); routes {"local": 1790, "llm_fallback": 388, "llm_failed": 1, "rejected_no_fallback": 2}; source: X-OpenAI-Usage provider_called per request

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| רכב | 704 | 0.9929 | 0.8963 | 0.8594 |
| נדל״ן | 657 | 0.9939 | 0.9361 | 0.8767 |
| יד_שנייה | 720 | 0.9931 | 0.9431 | 0.9194 |
| None | 100 | 0.9800 | 0.9800 | 0.9800 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 751 | 0.9920 | 0.9401 | 0.9068 |
| slang | 717 | 0.9916 | 0.9358 | 0.9149 |
| typo | 710 | 0.9944 | 0.9070 | 0.8479 |
| typo+standard | 3 | 1.0000 | 0.6667 | 0.6667 |

## By route

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| local | 1790 | 0.9994 | 0.9626 | 0.9296 |
| llm_fallback | 388 | 0.9691 | 0.7732 | 0.7139 |
| rejected_no_fallback | 2 | 0.0000 | 0.0000 | 0.0000 |
| llm_failed | 1 | 0.0000 | 0.0000 | 0.0000 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "f83bb4a1e0f009427b0689869d0a45ed915cee1adb8d6a40496e812751915f24", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": 0.050227245, "status": "estimated", "reported_usage_only": true, "has_failed_requests": true, "billable_tokens": {"input": 1167, "output": 59514, "cached_input": 1807692, "cache_write_30m": 18213, "cache_write_12h": 0}, "breakdown_usd": {"input": 0.0001167, "output": 0.029757, "cached_input": 0.01807692, "cache_write_30m": 0.002276625, "cache_write_12h": 0.0}, "estimated_per_query_usd": 2.302945667125172e-05, "projected_10m_queries_usd": 230.2945667125172, "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}`

Routing: `{"{\"destination\": \"local\", \"reason\": \"accepted\"}": 1697.0, "{\"destination\": \"fallback\", \"reason\": \"low_confidence\"}": 329.0, "{\"destination\": \"fallback\", \"reason\": \"empty_extraction\"}": 62.0, "{\"destination\": \"local\", \"reason\": \"non_search\"}": 93.0}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2181.0}`

Deployment: `{"image": "yad2-eval:e8bc46fb4a17", "image_id": "sha256:bbf75cf070f273cc2ac6907ad6b320f23c0a71fb66d61d076aba47d0013150a9", "container": "yad2-eval-e8bc46fb4a17", "runtime": {"PARSER_MODE": "combined", "CACHE_MAX_SIZE": "0", "EXPOSE_TOKEN_USAGE": "true", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "0", "OPENAI_GUIDELINES_ENABLED": "true", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "LOCAL_PARSER": "tagger", "TAGGER_HYBRID_VEHICLE_TYPE": "false", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "low", "E5_MODEL_PATH": "/models/e5", "CLASSIFIER_ARTIFACT_PATH": "/app/src/artifacts/classifier.joblib", "CONFIDENCE_ARTIFACT_PATH": "/app/src/artifacts/confidence.joblib", "TAGGER_MODEL_DIR": "/models/tagger"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 1 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.
