# Parser evaluation

Mode: api; split: test; model: gpt-6-luna; effort: low

| Metric | Value |
| --- | --- |
| rows | 1 |
| category_correct | 1.0 |
| field_set_exact | 0.0 |
| json_exact | 0.0 |
| failures | {} |
| latency_p50_seconds | 4.1545848999521695 |
| latency_p95_seconds | 4.1545848999521695 |
| successful_latency_p95_seconds | 4.1545848999521695 |
| tokens | {"input": 4561.0, "output": 175.0, "cached_input": 4558.0, "reasoning": 116.0} |

## By category

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| רכב | 1 | 1.0000 | 0.0000 | 0.0000 |

## By language

| Slice | Rows | Category accuracy | Field set exact | JSON exact |
| --- | --- | --- | --- | --- |
| standard | 1 | 1.0000 | 0.0000 | 0.0000 |

Routing: `{}`

Cache counter changes: `{"parse_cache_hits_total": 0.0, "parse_cache_misses_total": 1.0}`

Deployment: `{"image": "yad2-eval:8c4015c5d1ab", "image_id": "sha256:a86ca70705c690c7306371441adfce4df9e25a4190f6c2ff905863191a6c74a2", "container": "yad2-eval-8c4015c5d1ab", "runtime": {"PARSER_MODE": "openai", "CACHE_MAX_SIZE": "0", "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false", "EXTRACTOR_USE_TAGGER": "true", "OPENAI_TIMEOUT_SECONDS": "30.0", "OPENAI_MAX_OUTPUT_TOKENS": "4096", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "low"}, "cache_capacity": 0, "transport": "HTTP POST /parse to Docker published localhost port"}`

Latency scope: client-observed POST /parse over HTTP, sequential requests.
Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.
Token breakdowns overlap: cached input is included in input; reasoning is included in output.
Per-query records contain dataset query text and predictions.
