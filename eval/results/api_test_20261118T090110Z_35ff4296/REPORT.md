# Parser evaluation

Mode: api; split: test; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 2181 |
| category_correct | 0.959651535992664 |
| field_set_exact | 0.889958734525447 |
| json_exact | 0.8271435121503897 |
| failures | {"http_503": 83} |
| latency_p50_seconds | 0.06537480000406504 |
| latency_p95_seconds | 0.092700999986846 |
| successful_latency_p95_seconds | 0.09282372502202633 |
| tokens | {} |

LLM fallback: 0 of 2181 requests (0.0%); routes {"local": 2098, "rejected_no_fallback": 83}; source: X-OpenAI-Usage provider_called per request

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| רכב | 704 | 0.9688 | 0.8949 | 0.8310 |
| נדל״ן | 657 | 0.9635 | 0.8798 | 0.7823 |
| יד_שנייה | 720 | 0.9514 | 0.8889 | 0.8500 |
| None | 100 | 0.9300 | 0.9300 | 0.9300 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 751 | 0.9627 | 0.9041 | 0.8575 |
| slang | 717 | 0.9637 | 0.9052 | 0.8536 |
| typo | 710 | 0.9521 | 0.8592 | 0.7690 |
| typo+standard | 3 | 1.0000 | 1.0000 | 0.6667 |

## By route

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| local | 2098 | 0.9976 | 0.9252 | 0.8599 |
| rejected_no_fallback | 83 | 0.0000 | 0.0000 | 0.0000 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "f83bb4a1e0f009427b0689869d0a45ed915cee1adb8d6a40496e812751915f24", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": null, "status": "unavailable", "reason": "token_usage_not_reported"}`

Routing: `{"{\"destination\": \"local\", \"reason\": \"accepted\"}": 2026.0, "{\"destination\": \"fallback\", \"reason\": \"empty_extraction\"}": 62.0, "{\"destination\": \"local\", \"reason\": \"non_search\"}": 93.0}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2181.0}`

Deployment: `{"image": "yad2-eval:8b46a514cb9f", "image_id": "sha256:977a7a865807999b2e67eadb03ae3a2c7f9f53a8f569af9a0c141a1d63ff44d9", "container": "yad2-eval-8b46a514cb9f", "runtime": {"PARSER_MODE": "local", "CACHE_MAX_SIZE": "0", "EXPOSE_TOKEN_USAGE": "true", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "0", "OPENAI_GUIDELINES_ENABLED": "true", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "LOCAL_PARSER": "tagger", "TAGGER_HYBRID_VEHICLE_TYPE": "false", "PARSE_CONFIDENCE_THRESHOLD": "0.0", "E5_MODEL_PATH": "/models/e5", "CLASSIFIER_ARTIFACT_PATH": "/app/src/artifacts/classifier.joblib", "CONFIDENCE_ARTIFACT_PATH": "/app/src/artifacts/confidence.joblib", "TAGGER_MODEL_DIR": "/models/tagger"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 1 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.
