# Parser evaluation

Mode: api; split: val; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 2042 |
| category_correct | 0.9946131243878551 |
| field_set_exact | 0.9255631733594515 |
| json_exact | 0.9142997061704211 |
| failures | {"http_502": 2} |
| latency_p50_seconds | 2.0547768999822438 |
| latency_p95_seconds | 3.5317572350060793 |
| successful_latency_p95_seconds | 3.532088700006716 |
| tokens | {"input": 10635421.0, "output": 219717.0, "cached_input": 10546650.0, "reasoning": 117474.0, "cache_write_30m": 82645.0, "cache_write_12h": 0.0} |

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| יד_שנייה | 638 | 0.9922 | 0.8824 | 0.8777 |
| רכב | 647 | 0.9969 | 0.9335 | 0.9274 |
| נדל״ן | 657 | 0.9939 | 0.9482 | 0.9239 |
| None | 100 | 1.0000 | 1.0000 | 1.0000 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 680 | 0.9971 | 0.9309 | 0.9176 |
| typo | 677 | 0.9970 | 0.9158 | 0.8981 |
| slang | 685 | 0.9898 | 0.9299 | 0.9270 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "e591aead0eaad6e7457befcdc6946a7448879746ae3190c1fe625de4db3cfea7", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": 0.226268225, "status": "estimated", "reported_usage_only": true, "has_failed_requests": true, "billable_tokens": {"input": 6126, "output": 219717, "cached_input": 10546650, "cache_write_30m": 82645, "cache_write_12h": 0}, "breakdown_usd": {"input": 0.0006126, "output": 0.1098585, "cached_input": 0.1054665, "cache_write_30m": 0.010330625, "cache_write_12h": 0.0}, "estimated_per_query_usd": 0.00011080716209598432, "projected_10m_queries_usd": 1108.0716209598434, "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}`

Routing: `{}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2042.0}`

Deployment: `{"image": "yad2-eval:307eeb37b8f9", "image_id": "sha256:29aca2ea5d02fe76c159b2fa632952418f9326e93f907c9d5024d5f8ec036e9c", "container": "yad2-eval-307eeb37b8f9", "runtime": {"PARSER_MODE": "openai", "CACHE_MAX_SIZE": "0", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "12", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "low"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 8 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.
