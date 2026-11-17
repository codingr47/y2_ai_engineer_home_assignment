# Parser evaluation

Mode: api; split: val; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 2042 |
| category_correct | 0.990205680705191 |
| field_set_exact | 0.925073457394711 |
| json_exact | 0.9098922624877571 |
| failures | {"http_502": 1} |
| latency_p50_seconds | 1.9715507500222884 |
| latency_p95_seconds | 3.2338684649876077 |
| successful_latency_p95_seconds | 3.2341440999880433 |
| tokens | {"input": 9969729.0, "output": 231345.0, "cached_input": 9882262.0, "reasoning": 129023.0, "cache_write_30m": 81341.0, "cache_write_12h": 0.0} |

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| יד_שנייה | 638 | 0.9875 | 0.9060 | 0.8856 |
| רכב | 647 | 0.9923 | 0.9073 | 0.9011 |
| נדל״ן | 657 | 0.9893 | 0.9498 | 0.9285 |
| None | 100 | 1.0000 | 1.0000 | 1.0000 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 680 | 0.9941 | 0.9382 | 0.9221 |
| typo | 677 | 0.9956 | 0.9188 | 0.9025 |
| slang | 685 | 0.9810 | 0.9182 | 0.9051 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "e591aead0eaad6e7457befcdc6946a7448879746ae3190c1fe625de4db3cfea7", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": 0.225275345, "status": "estimated", "reported_usage_only": true, "has_failed_requests": true, "billable_tokens": {"input": 6126, "output": 231345, "cached_input": 9882262, "cache_write_30m": 81341, "cache_write_12h": 0}, "breakdown_usd": {"input": 0.0006126, "output": 0.1156725, "cached_input": 0.09882262, "cache_write_30m": 0.010167625, "cache_write_12h": 0.0}, "estimated_per_query_usd": 0.00011032093290891283, "projected_10m_queries_usd": 1103.2093290891282, "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}`

Routing: `{}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2042.0}`

Deployment: `{"image": "yad2-eval:153630f93188", "image_id": "sha256:aca0a2bbef252750e7bebe840e39d22e592d72e07b842fafa27d10b9c93d458b", "container": "yad2-eval-153630f93188", "runtime": {"PARSER_MODE": "openai", "CACHE_MAX_SIZE": "0", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "6", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "low"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 8 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.
