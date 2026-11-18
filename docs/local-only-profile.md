# Local-only profile: meeting the latency target without the LLM

The shipped profile (tagger + gate + Luna fallback) is the most accurate cost-efficient option, at **89.00%**, but its
whole-pipeline **p95 is 3.00 s**, five times the 600 ms target. The 18% of queries sent to the LLM take 2.5 s at the
median, all provider latency, and no threshold fixes that while they still go to the LLM.

The local-only profile drops the LLM. **Every query is answered by the tagger, p95 is 93 ms, and there is no LLM bill.**
The price is accuracy: **82.71% exact overall**, and **14.0% of the answers it returns are not exact**.

## Configuration

```dotenv
PARSER_MODE=local
LOCAL_PARSER=tagger
EXTRACTOR_ENABLED=true
EXTRACTOR_USE_TAGGER=true
# Turns the gate off: every local answer is returned. Without it, local mode returns 503 for the ~18% the gate rejects.
PARSE_CONFIDENCE_THRESHOLD=0
TAGGER_HYBRID_VEHICLE_TYPE=false
CACHE_MAX_SIZE=1024
```

- **No OpenAI key is needed.**
- **The gate still runs, but only to score.** `confidence` in each response is still the gate's estimate that the
  whole parse is right, so clients can use it, for example to show low-confidence filters as suggestions.
- **The gate's audited operating point doesn't apply.** At threshold 0 nothing is rejected, so its 95% claim is void.

## Measured result

One Docker HTTP run of all 2,181 test queries, one client, cache off, on the same code as `main`. Report:
[md](../eval/results/api_test_20261118T090110Z_35ff4296/REPORT.md) ·
[html](../eval/results/api_test_20261118T090110Z_35ff4296/REPORT.html).

| | **local only** | shipped (gate + Luna fallback) |
| --- | --- | --- |
| JSON exact | **82.71%** | 89.00% |
| field set exact | 89.00% | 92.76% |
| category correct | 95.97% | 99.27% |
| answered | 96.2% (83 × HTTP 503) | 99.9% |
| latency p50 / p95 / p99 | **65 / 93 / 113 ms** (max 196 ms) | 74 ms / 3.00 s / 4.39 s |
| LLM cost per 10M queries | **$0** | $230 |

**Under load,** the local path holds **12 QPS per container with p95 179 ms**
([docs/benchmark_local.json](benchmark_local.json)), so the target is met with headroom at the container's ceiling.

### Where the 17.3% goes

| outcome | queries | share of all |
| --- | --- | --- |
| exact | 1,804 | 82.7% |
| answered, wrong value (right fields) | 137 | 6.3% |
| answered, wrong field set (a field missed or added) | 152 | 7.0% |
| answered, wrong category | 5 | 0.2% |
| **not answered** (HTTP 503) | 83 | 3.8% |

- **Most wrong answers are partial filters, not wrong searches.** Among answered queries the category is right
  99.8% of the time.
  - Of the field-set errors, 108 miss a field the user asked for, and 70 add one they didn't. Some do both.
- **The 503s:** 62 queries where the tagger extracts nothing, and 21 local answers that fail the taxonomy check.
  - Without a fallback, the service has no answer for them.
  - A client should treat `503 fallback_required` as "no structured filters" and fall back to a plain text search.

### By vertical

| vertical | queries | exact | not answered |
| --- | --- | --- | --- |
| רכב | 704 | 83.1% | 21 |
| נדל״ן | 657 | **78.2%** | 22 |
| יד_שנייה | 720 | 85.0% | 34 |
| not a search | 100 | 93.0% | 6 |

Real estate loses the most against the shipped profile (87.7% → 78.2%), vehicles the least (85.9% → 83.1%).

## The trade-off

Compared with the shipped profile on the same queries, 162 answers get worse and 25 get better: **−6.3 points**.
- 108 are queries where the LLM was right and the local answer isn't.
- 54 are queries the LLM answered correctly that now get a 503.

| | local only | shipped |
| --- | --- | --- |
| meets p95 ≤ 600 ms | **yes** (93 ms; 179 ms at 12 QPS) | no (3.00 s) |
| exact answers | 82.7% | 89.0% |
| failure mode | partial filters, 3.8% unanswered | slow tail on 18% of queries |
| external dependency | none | OpenAI availability, keys, budget |
| cost per 10M queries a month | **about $220–290** (hosting only) | about $450–520 |

Hosting is the same assumption as in the [README](../README.md#estimate-for-10m-queries-a-month): two 4 vCPU / 8 GB
containers.

**It's a viable trade-off when the latency requirement is firm.** An inexact answer is usually a partial filter on the
right vertical, which the user can refine. The shipped profile is still the better choice when accuracy matters more
than the tail latency.

## Caveats

- **One run on the test set,** which has been inspected during development: regression evidence, not an untouched
  estimate. The local answers are deterministic, so a rerun gives the same accuracy.
- **The offline harness reports 85.1% for the same tagger.** The HTTP figure is lower mainly because it counts the
  83 unanswered queries as wrong.
- **A nearby option, not measured live:** threshold 0, keeping the LLM only for the 3.8% that have no local answer.
  Replayed from recorded runs on `exp/lower-threshold`, it gives about 85.2% at p95 108 ms and about $43 per 10M
  queries. That's 2.5 points more and no 503s, but it keeps the OpenAI dependency.
