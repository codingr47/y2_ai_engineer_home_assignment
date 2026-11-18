# Local-only profile: meeting the latency target without the LLM

The shipped profile (tagger + gate + Luna fallback) is the most accurate cost-efficient option, at **90.05%**, but its
whole-pipeline **p95 is 3.13 s**, five times the 600 ms target. The 16% of queries sent to the LLM take 2.8 s at the
median, all provider latency, and no threshold fixes that while they still go to the LLM.

The local-only profile drops the LLM. **The tagger answers 98.9% of queries, p95 is 99 ms, and there is no LLM bill.**
The price is accuracy: **84.87% exact overall**, and **14.2% of the answers it returns are not exact**.

## Configuration

```dotenv
PARSER_MODE=local
LOCAL_PARSER=tagger
EXTRACTOR_ENABLED=true
EXTRACTOR_USE_TAGGER=true
# Turns the gate off: every local answer is returned. Without it, local mode returns 503 for the ~15% the gate rejects.
PARSE_CONFIDENCE_THRESHOLD=0
TAGGER_HYBRID_VEHICLE_TYPE=false
CACHE_MAX_SIZE=1024
```

- **No OpenAI key is needed.**
- **The gate still runs, but only to score.** `confidence` in each response is still the gate's estimate that the
  whole parse is right, so clients can use it, for example to show low-confidence filters as suggestions.
- **The gate's audited operating point doesn't apply.** At threshold 0 nothing is rejected, so its 95% claim is void.

## Measured result

One Docker HTTP run of all 2,181 test queries, one client, cache off. Report:
[md](../eval/results/api_test_20261118T141242Z_952b9b6b/REPORT.md) ·
[html](../eval/results/api_test_20261118T141242Z_952b9b6b/REPORT.html). The shipped column is
[api_test_20261118T141547Z_5c745d7c](../eval/results/api_test_20261118T141547Z_5c745d7c/REPORT.md), on the same code.

| | **local only** | shipped (gate + Luna fallback) |
| --- | --- | --- |
| JSON exact | **84.87%** | 90.05% |
| field set exact | 91.15% | 93.72% |
| category correct | 98.30% | 99.36% |
| answered | 98.9% (23 × HTTP 503) | 99.9% |
| latency p50 / p95 / p99 | **64 / 99 / 129 ms** (max 217 ms) | 69 ms / 3.13 s / 4.18 s |
| LLM cost per 10M queries | **$0** | $196 |

**Under load,** the local path held **12 QPS per container with p95 179 ms** ([docs/benchmark_local.json](benchmark_local.json)).
That benchmark predates answering filter-less searches locally, which adds a second inference on about 3% of queries,
so expect it slightly higher; it's still far below the target.

### Where the 15.1% goes

| outcome | queries | share of all |
| --- | --- | --- |
| exact | 1,851 | 84.9% |
| answered, wrong value (right fields) | 137 | 6.3% |
| answered, wrong field set (a field missed or added) | 156 | 7.2% |
| answered, wrong category | 14 | 0.6% |
| **not answered** (HTTP 503) | 23 | 1.1% |

- **Most wrong answers are partial filters, not wrong searches.** Among answered queries the category is right
  99.4% of the time.
  - Of the field-set errors, 112 miss a field the user asked for, and 70 add one they didn't. Some do both.
  - The wrong categories include 6 non-searches (for example "מה ההבדל בין טאבו לחכירה") read as a vertical with no
    filters, which then show unfiltered listings.
- **The 503s:** 21 local answers that fail the taxonomy check (17 of them second-hand items), and 2 queries that
  contain injection text and are blocked as before.
  - Without a fallback, the service has no answer for them.
  - A client should treat `503 fallback_required` as "no structured filters" and fall back to a plain text search.

### By vertical

| vertical | queries | exact | not answered | shipped |
| --- | --- | --- | --- | --- |
| רכב | 704 | 85.1% | 4 | 87.6% |
| נדל״ן | 657 | **81.0%** | 2 | 87.8% |
| יד_שנייה | 720 | 87.1% | 17 | 93.3% |
| not a search | 100 | 93.0% | 0 | 98.0% |

Real estate loses the most against the shipped profile, vehicles the least.

## Searches with no filters

A query like "מחפש בית יפה" (looking for a nice house) is a real-estate search with nothing filterable in it. The right
answer is `נדל״ן` with `{}`, and the gold labels say so.

- **Until now, the service never returned an answer with empty params:** the local parsers escalate it
  (`empty_extraction`) before the gate sees it. Without a fallback, that was a 503.
- **On the test set, the tagger is right on 55 of those 62 queries.** Luna, which they were escalated to, gets
  only 43 right. It tends to invent a filter ("a nice house" → `בית פרטי/וילה`).
- **They are now treated like any other local answer** (`src/parsing/empty_params.py`): returned when the gate
  accepts them. At threshold 0, that's always.
- **Injection text keeps its old handling.** A filter-less answer to a query the injection guard flags is still
  escalated, so it can't return the category an attack asked for.

**Effect, measured:** local-only 82.71% → **84.87%** (503s 83 → 23); shipped profile +0.46 points from the change,
with 49 of the 62 answered locally there.

## Why the offline harness shows 85.1%

The README's "raw predictions" row (85.14%) is the same tagger. The training harness scores every raw prediction
and refuses nothing. The HTTP run gets 6 fewer exact answers:
- **2 the service still refuses:** filter-less answers to queries with injection text, kept blocked on purpose.
- **6 the harness counts too generously:** non-searches the tagger reads as a vertical with empty params. The harness
  compares only the params (`{}` = `{}`); the HTTP evaluation also checks that the category is `null`.
- **2 the other way:** the serving normaliser fixes them (e.g. "ברחביה" → "רחביה").

1,857 − 2 − 6 + 2 = 1,851, the HTTP count.

## The trade-off

Compared with the shipped profile on the same queries, 132 answers get worse and 19 get better: **−5.2 points**.
- 119 are queries where the LLM was right and the local answer isn't.
- 13 are queries the LLM answered correctly that now get a 503.

| | local only | shipped |
| --- | --- | --- |
| meets p95 ≤ 600 ms | **yes** (99 ms) | no (3.13 s) |
| exact answers | 84.9% | 90.1% |
| failure mode | partial filters, 1.1% unanswered | slow tail on 16% of queries |
| external dependency | none | OpenAI availability, keys, budget |
| cost per 10M queries a month | **about $220–290** (hosting only) | about $420–490 |

Hosting is the same assumption as in the [README](../README.md#estimate-for-10m-queries-a-month): two 4 vCPU / 8 GB
containers.

**It's a viable trade-off when the latency requirement is firm.** An inexact answer is usually a partial filter on the
right vertical, which the user can refine. The shipped profile is still the better choice when accuracy matters more
than the tail latency.

## Caveats

- **One run on the test set,** which has been inspected during development: regression evidence, not an untouched
  estimate. The local answers are deterministic, so a rerun gives the same accuracy.
- **The shipped column has LLM noise:** Luna's answers vary by about ±0.5 points between runs.
- **A nearby option, if an OpenAI dependency is acceptable:** `PARSER_MODE=combined` with the same threshold 0. Only
  the 21 taxonomy-check failures reach Luna. Measured: **85.47%**, p95 99 ms, no 503s, about $12 per 10M queries
  ([report](../eval/results/api_test_20261118T144421Z_770b6f62/REPORT.md)). See the README's
  [configuration choice](../README.md#choosing-a-configuration-latency-accuracy-and-cost).
