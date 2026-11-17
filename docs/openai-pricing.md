# OpenAI evaluation cost estimates

The machine-readable lookup is `eval/openai_prices.json`. This snapshot was
verified on **2026-11-17**, using OpenAI's official model pages and
[pricing documentation](https://developers.openai.com/api/docs/pricing).
Prices below are **USD per million text tokens, Standard processing**, for
requests with at most 272,000 input tokens.

| Model | Input | Cached input | Cache write, 30m | Cache write, 12h | Output |
| --- | ---: | ---: | ---: | ---: | ---: |
| [gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna) | $0.10 | $0.01 | $0.125 | $0.25 | $0.50 |
| [gpt-6.1-terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra) | $1.00 | $0.05 | $1.25 | $2.50 | $5.00 |
| [gpt-6.1-sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) | $2.00 | $0.10 | $2.50 | $5.00 | $10.00 |
| [gpt-6-sol](https://developers.openai.com/api/docs/models/gpt-6-sol) | $2.00 | $0.20 | $2.50 | $5.00 | $10.00 |
| [gpt-6-astra](https://developers.openai.com/api/docs/models/gpt-6-astra) | $10.00 | $1.00 | $12.50 | $25.00 | $50.00 |

Reports retain the price snapshot version, SHA-256, source, and exact applied
rates, so estimates remain reproducible after the lookup is updated. Unknown
model IDs produce an unavailable estimate rather than borrowing another price.
Verify and add explicit entries when testing additional models or snapshots.

## Calculation

Input usage includes cached input and cache writes. To avoid double counting,
the estimator splits it into disjoint buckets:

```text
uncached_input = total_input - cached_input - writes_30m - writes_12h
cost = (uncached_input * input_rate
      + cached_input * cached_rate
      + writes_30m * write_30m_rate
      + writes_12h * write_12h_rate
      + total_output * output_rate) / 1_000_000
```

Reasoning tokens are already included in output usage and are not added again.
Unclassified cache writes use the 30-minute rate. Rates assume Standard, short
context requests; this extractor's small fixed catalog and bounded queries fit
that scope. Long-context requests, other service tiers, regional premiums,
gateway markups, taxes, discounts, infrastructure, and tool fees are excluded.

HTTP evaluations read server token counters before and after the dataset run.
This counts reported usage even when generated output fails validation or a
request is refused. Errors without reported usage cannot be priced and may
leave an incomplete estimate. Missing usage is shown as unavailable, not free.
The isolated Docker container prevents unrelated traffic from entering counters.

Docker evaluations also enable `EXPOSE_TOKEN_USAGE=true`. Each HTTP response then
includes request-scoped `X-OpenAI-Usage` metadata, including reported usage for
rejected outputs. Concurrent requests are isolated. The normal response body is
unchanged, and usage headers are disabled by default outside evaluation.

Every new JSONL record stores its token usage, `estimated_cost_usd`, cost status,
and token-type cost breakdown. The HTML query table has an estimated-cost column;
expand a row to inspect its usage and calculation. Cache hits and local-only
responses have zero provider cost. Calls whose usage is unavailable show
unavailable rather than zero. Reports include cost coverage and the sum of priced
requests for reconciliation with the aggregate token estimate. Older HTTP runs
that saved only aggregate usage cannot be backfilled with actual per-query costs.

HTML, Markdown, and JSON reports include total estimated token cost, average cost
per dataset query, a token-type breakdown, and a projection for 10 million queries
with the **same query mix, routing, caching, token usage, and prices**. This is not
an invoice. In combined mode, dividing total token cost by all dataset queries
accounts for locally handled requests as part of the observed query mix.

To backfill cost and HTML for already completed runs without spending API tokens:

```powershell
python -m scripts.evaluation_html eval/results
```

Existing cost snapshots are preserved when regenerating HTML.
