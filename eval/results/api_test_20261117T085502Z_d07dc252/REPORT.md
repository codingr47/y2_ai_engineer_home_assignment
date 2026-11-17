# Parser evaluation

Mode: api; split: test; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 2181 |
| category_correct | 0.9862448418156808 |
| field_set_exact | 0.8546538285190279 |
| json_exact | 0.8225584594222833 |
| failures | {"http_502": 4} |
| latency_p50_seconds | 2.307996599993203 |
| latency_p95_seconds | 3.909717599977739 |
| successful_latency_p95_seconds | 3.9049461600137865 |
| tokens | {"input": 9951072.0, "output": 270267.0, "cached_input": 9873418.0, "reasoning": 159901.0, "cache_write_30m": 71111.0, "cache_write_12h": 0.0} |

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| רכב | 704 | 0.9773 | 0.7301 | 0.7202 |
| נדל״ן | 657 | 0.9924 | 0.9178 | 0.8432 |
| יד_שנייה | 720 | 0.9889 | 0.9000 | 0.8806 |
| None | 100 | 0.9900 | 0.9900 | 0.9900 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 751 | 0.9880 | 0.8575 | 0.8282 |
| slang | 717 | 0.9902 | 0.8508 | 0.8173 |
| typo | 710 | 0.9803 | 0.8563 | 0.8225 |
| typo+standard | 3 | 1.0000 | 0.6667 | 0.6667 |

Estimated token cost: `{"currency": "USD", "model": "gpt-6-luna", "price_version": "2026-11-17", "verified_on": "2026-11-17", "price_table_sha256": "e591aead0eaad6e7457befcdc6946a7448879746ae3190c1fe625de4db3cfea7", "service_tier": "standard", "assumptions": ["Standard processing, text tokens, no more than 272000 input tokens per request.", "Excludes taxes, infrastructure, regional premiums, gateway markups, and negotiated discounts.", "Reasoning tokens are already included in output tokens and are not charged twice.", "Cache writes with no retention breakdown use the 30-minute write rate.", "Costs cover reported token usage only; failures without usage may leave an incomplete estimate."], "source": "https://developers.openai.com/api/docs/models/gpt-6-luna", "rates_per_million_tokens": {"input": 0.1, "cached_input": 0.01, "cache_write_30m": 0.125, "cache_write_12h": 0.25, "output": 0.5, "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"}, "estimated_total_usd": 0.243410855, "status": "estimated", "reported_usage_only": true, "has_failed_requests": true, "billable_tokens": {"input": 6543, "output": 270267, "cached_input": 9873418, "cache_write_30m": 71111, "cache_write_12h": 0}, "breakdown_usd": {"input": 0.0006543, "output": 0.1351335, "cached_input": 0.09873418, "cache_write_30m": 0.008888875, "cache_write_12h": 0.0}, "estimated_per_query_usd": 0.00011160516047684549, "projected_10m_queries_usd": 1116.0516047684548, "projection_assumption": "Same query mix, routing, cache behavior, token usage, and prices as this run."}`

Routing: `{}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 2181.0}`

Deployment: `{"image": "yad2-eval:380533fab862", "image_id": "sha256:499f9cd71e09a401e100e219f09f1a18db9c94c294fa477a8115b8be60e34abf", "container": "yad2-eval-380533fab862", "runtime": {"PARSER_MODE": "openai", "CACHE_MAX_SIZE": "0", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "low"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, 8 concurrent clients; excludes client queue wait.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.
