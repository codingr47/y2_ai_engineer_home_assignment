# Parser evaluation

Mode: api; split: val; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 2042 |
| category_correct | 0.9833496571988247 |
| field_set_exact | 0.876101860920666 |
| json_exact | 0.8491674828599413 |
| failures | {"http_502": 6} |
| latency_p50_seconds | 2.1149813999945763 |
| latency_p95_seconds | 3.5532526899594794 |
| successful_latency_p95_seconds | 3.5568291749659693 |
| tokens | {"input": 9312205.0, "output": 244791.0, "cached_input": 9244134.0, "reasoning": 141667.0, "cache_write_30m": 61945.0, "cache_write_12h": 0.0} |

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| יד_שנייה | 638 | 0.9765 | 0.8762 | 0.8558 |
| רכב | 647 | 0.9861 | 0.8053 | 0.7975 |
| נדל״ן | 657 | 0.9863 | 0.9285 | 0.8721 |
| None | 100 | 0.9900 | 0.9900 | 0.9900 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 680 | 0.9838 | 0.8824 | 0.8559 |
| typo | 677 | 0.9838 | 0.8685 | 0.8405 |
| slang | 685 | 0.9825 | 0.8774 | 0.8511 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "e591aead0eaad6e7457befcdc6946a7448879746ae3190c1fe625de4db3cfea7", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": 0.223192565, "status": "estimated", "reported_usage_only": true, "has_failed_requests": true, "billable_tokens": {"input": 6126, "output": 244791, "cached_input": 9244134, "cache_write_30m": 61945, "cache_write_12h": 0}, "breakdown_usd": {"input": 0.0006126, "output": 0.1223955, "cached_input": 0.09244134, "cache_write_30m": 0.007743125, "cache_write_12h": 0.0}, "estimated_per_query_usd": 0.00010930096229187071, "projected_10m_queries_usd": 1093.0096229187072, "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}`

Routing: `{}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2042.0}`

Deployment: `{"image": "yad2-eval:a1a96266b048", "image_id": "sha256:74b8d277b2eeb99d04249f540e55ab14474c1f37a28f2e2476180b1c68feaf7d", "container": "yad2-eval-a1a96266b048", "runtime": {"PARSER_MODE": "openai", "CACHE_MAX_SIZE": "0", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_FEW_SHOT_COUNT": "0", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "low"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 8 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.
