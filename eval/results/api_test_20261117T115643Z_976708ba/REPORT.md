# Parser evaluation

Mode: api; split: test; model: gpt-6.1-terra; effort: low

| Metric | Value |
| --- | --- |
| rows | 2181 |
| category_correct | 0.9917469050894085 |
| field_set_exact | 0.9509399358092618 |
| json_exact | 0.9362677670793215 |
| failures | {"http_502": 4} |
| latency_p50_seconds | 2.1159785000490956 |
| latency_p95_seconds | 3.5626550000160933 |
| successful_latency_p95_seconds | 3.564207180019001 |
| tokens | {"input": 11530116.0, "output": 119769.0, "cached_input": 11441929.0, "reasoning": 14466.0, "cache_write_30m": 81644.0, "cache_write_12h": 0.0} |

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| רכב | 704 | 0.9957 | 0.9474 | 0.9432 |
| נדל״ן | 657 | 0.9939 | 0.9422 | 0.9026 |
| יד_שנייה | 720 | 0.9875 | 0.9583 | 0.9542 |
| None | 100 | 0.9800 | 0.9800 | 0.9800 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 751 | 0.9947 | 0.9507 | 0.9348 |
| slang | 717 | 0.9916 | 0.9554 | 0.9442 |
| typo | 710 | 0.9887 | 0.9479 | 0.9310 |
| typo+standard | 3 | 1.0000 | 0.6667 | 0.6667 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6.1-terra", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "e591aead0eaad6e7457befcdc6946a7448879746ae3190c1fe625de4db3cfea7", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6.1-terra", "rates_per_million_tokens": {"input": 1.0, "cached_input": 0.05, "cache_write_30m": 1.25, "cache_write_12h": 2.5, "output": 5.0, "source": "https://developers.openai.com/api/docs/models/gpt-6.1-terra"}, "estimated_total_usd": 1.27953945, "status": "estimated", "reported_usage_only": true, "has_failed_requests": true, "billable_tokens": {"input": 6543, "output": 119769, "cached_input": 11441929, "cache_write_30m": 81644, "cache_write_12h": 0}, "breakdown_usd": {"input": 0.006543, "output": 0.598845, "cached_input": 0.57209645, "cache_write_30m": 0.102055, "cache_write_12h": 0.0}, "estimated_per_query_usd": 0.0005866755845942228, "projected_10m_queries_usd": 5866.755845942228, "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}`

Routing: `{}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2181.0}`

Deployment: `{"image": "yad2-eval:dc49d29eeceb", "image_id": "sha256:1ba08974063ba6e0b8d63528bb2f31a43f65ce61b4037e5cd48f5f0bf24fd68e", "container": "yad2-eval-dc49d29eeceb", "runtime": {"PARSER_MODE": "openai", "CACHE_MAX_SIZE": "0", "EXPOSE_TOKEN_USAGE": "true", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "12", "OPENAI_GUIDELINES_ENABLED": "true", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "OPENAI_MODEL": "gpt-6.1-terra", "OPENAI_REASONING_EFFORT": "low"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 8 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.

Evaluation role: exposed regression; test failures informed RCA, so this is not an independent accuracy claim.
