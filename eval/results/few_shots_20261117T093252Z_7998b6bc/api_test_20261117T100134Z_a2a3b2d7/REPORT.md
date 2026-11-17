# Parser evaluation

Mode: api; split: test; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 2181 |
| category_correct | 0.9944979367262724 |
| field_set_exact | 0.9055479138010087 |
| json_exact | 0.8895002292526364 |
| failures | {"http_502": 4} |
| latency_p50_seconds | 2.0836353999911807 |
| latency_p95_seconds | 3.416687600023579 |
| successful_latency_p95_seconds | 3.4170324600185262 |
| tokens | {"input": 11364360.0, "output": 246153.0, "cached_input": 11286675.0, "reasoning": 136680.0, "cache_write_30m": 71142.0, "cache_write_12h": 0.0} |

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| רכב | 704 | 0.9943 | 0.8750 | 0.8693 |
| נדל״ן | 657 | 0.9970 | 0.9437 | 0.9072 |
| יד_שנייה | 720 | 0.9958 | 0.8917 | 0.8819 |
| None | 100 | 0.9700 | 0.9700 | 0.9700 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 751 | 0.9960 | 0.9134 | 0.8988 |
| slang | 717 | 0.9902 | 0.9107 | 0.8912 |
| typo | 710 | 0.9972 | 0.8930 | 0.8789 |
| typo+standard | 3 | 1.0000 | 0.6667 | 0.6667 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "e591aead0eaad6e7457befcdc6946a7448879746ae3190c1fe625de4db3cfea7", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": 0.2454903, "status": "estimated", "reported_usage_only": true, "has_failed_requests": true, "billable_tokens": {"input": 6543, "output": 246153, "cached_input": 11286675, "cache_write_30m": 71142, "cache_write_12h": 0}, "breakdown_usd": {"input": 0.0006543, "output": 0.1230765, "cached_input": 0.11286675, "cache_write_30m": 0.00889275, "cache_write_12h": 0.0}, "estimated_per_query_usd": 0.0001125585969738652, "projected_10m_queries_usd": 1125.585969738652, "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}`

Routing: `{}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2181.0}`

Deployment: `{"image": "yad2-eval:2bf7b585b892", "image_id": "sha256:ca373f1f80e5f0ad31ebde4ebbe952469361b7af80de87b1ddf1c0ee2e0befe8", "container": "yad2-eval-2bf7b585b892", "runtime": {"PARSER_MODE": "openai", "CACHE_MAX_SIZE": "0", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "12", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "low"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 8 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.

Evaluation role: exposed regression; test failures informed RCA, so this is not an independent accuracy claim.
