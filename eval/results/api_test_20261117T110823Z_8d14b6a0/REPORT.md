# Parser evaluation

Mode: api; split: test; model: gpt-6-luna; effort: medium

| Metric | Value |
| --- | --- |
| rows | 2181 |
| category_correct | 0.9940394314534617 |
| field_set_exact | 0.9188445667125172 |
| json_exact | 0.9023383768913342 |
| failures | {"http_503": 1, "http_502": 2} |
| latency_p50_seconds | 2.17626510001719 |
| latency_p95_seconds | 4.004803400021046 |
| successful_latency_p95_seconds | 4.005362105014502 |
| tokens | {"input": 11524824.0, "output": 322740.0, "cached_input": 11441929.0, "reasoning": 212959.0, "cache_write_30m": 76355.0, "cache_write_12h": 0.0} |

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| רכב | 704 | 0.9943 | 0.8722 | 0.8651 |
| נדל״ן | 657 | 0.9954 | 0.9406 | 0.9041 |
| יד_שנייה | 720 | 0.9944 | 0.9361 | 0.9264 |
| None | 100 | 0.9800 | 0.9800 | 0.9800 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 751 | 0.9947 | 0.9161 | 0.9015 |
| slang | 717 | 0.9944 | 0.9303 | 0.9149 |
| typo | 710 | 0.9930 | 0.9113 | 0.8915 |
| typo+standard | 3 | 1.0000 | 0.6667 | 0.6667 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "e591aead0eaad6e7457befcdc6946a7448879746ae3190c1fe625de4db3cfea7", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": 0.285987665, "status": "estimated", "reported_usage_only": true, "has_failed_requests": true, "billable_tokens": {"input": 6540, "output": 322740, "cached_input": 11441929, "cache_write_30m": 76355, "cache_write_12h": 0}, "breakdown_usd": {"input": 0.000654, "output": 0.16137, "cached_input": 0.11441929, "cache_write_30m": 0.009544375, "cache_write_12h": 0.0}, "estimated_per_query_usd": 0.00013112685236130216, "projected_10m_queries_usd": 1311.2685236130214, "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}`

Routing: `{}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2181.0}`

Deployment: `{"image": "yad2-eval:7b2388ab86f0", "image_id": "sha256:aef8b9c855f19d4b5a8327437277213f5d8f977c44382cb2a3296a5ee4696bb7", "container": "yad2-eval-7b2388ab86f0", "runtime": {"PARSER_MODE": "openai", "CACHE_MAX_SIZE": "0", "EXPOSE_TOKEN_USAGE": "true", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "12", "OPENAI_GUIDELINES_ENABLED": "true", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "medium"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 8 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.

Evaluation role: exposed regression; test failures informed RCA, so this is not an independent accuracy claim.
