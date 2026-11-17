# Yad2 - AI Engineer Home Assignment

## Running the API

By default the API returns **one fixed stub response for every valid query**.
Set `EXTRACTOR_ENABLED=true` in `.env` to use Exp3 fusion classification and the
value extractor. A confidence gate routes uncertain queries to an injectable
fallback; without one, they return HTTP 503. An optional OpenAI fallback supports
configurable model and reasoning effort. See [OpenAI configuration and evaluation reports](docs/openai-fallback.md)
and [local assets and calibration](docs/optional-parser.md).
Add `LOCAL_PARSER=tagger` to serve the fine-tuned DictaBERT tagger as the local parser (85.1% JSON
exact on test against 75.4%; its gate answers about 85% of queries locally, and the rest go to the
fallback, e.g. OpenAI in `combined` mode). It needs provisioned weights, see
[the tagger parser](docs/tagger-parser.md).
The disabled-mode response has category `נדל״ן`, the apartment filters from
Example 1, `confidence: 0.0`, and a stub note.

```sh
python -m pip install -r requirements-dev.txt
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Swagger UI: http://127.0.0.1:8000/docs. OpenAPI: `/openapi.json`.

| Endpoint | Behavior |
| --- | --- |
| `POST /parse` | Accepts `{"q":"דירת 3 חדרים בירושלים עד מליון שח"}`; returns the fixed response below. |
| `GET /health` | Returns `{"status":"ok"}` with HTTP 200; checks service liveness. |
| `GET /metrics` | Returns Prometheus text metrics from this application instance. |

```json
{
  "category": "נדל״ן",
  "params": {
    "סוגי_נכס": ["דירה"],
    "מס׳_חדרים": 3,
    "עיר": "ירושלים",
    "מחיר": {"min": 0, "max": 1000000}
  },
  "confidence": 0.0,
  "notes": ["Stub response; query interpretation is not implemented."]
}
```

Requests must contain only `q`, a nonblank string of at most 2,000 characters
before trimming. Invalid requests return HTTP 422. Surrounding whitespace is
trimmed; all other content is preserved. Hebrew interpretation, language
validation and typo correction remain limited to the extractor's existing behavior.
Enabled parsing validates fields and values against its versioned extraction schema.
Unexpected parser failures return HTTP 500 with `{"detail":"Internal server error"}`.
OpenAI fallback returns `category: null` and empty `params` for non-search queries;
its numeric confidence is an uncalibrated model self-rating, identified in `notes`.
Provider failures return sanitized HTTP 502/503 errors and are never cached.
The public vehicle contract uses `יצרן` and `דגם`, mapping to the taxonomy and
dataset editor's `יצרנים` and `דגמים`; the dataset tooling retains its existing keys.

### Architecture and cache

FastAPI validates HTTP contracts and delegates cache misses to an asynchronous
`Parser` interface. `StubParser` constructs the fixed fixture using typed parameter
models. The local parser replaces it when enabled without changing the route or
cache. The response model validates the envelope; the local parser validates
category-specific fields, value types, bounds, and canonical enum values.

The process-local, thread-safe LRU stores successful responses by trimmed query,
refreshes recency on a hit, and evicts the least recently used entry at capacity.
Stored and retrieved values are copied to prevent cross-request mutation. Failed
requests are never cached. Concurrent misses for the same key can invoke the
parser more than once; request coalescing is deferred until external model calls
are introduced.

`CACHE_MAX_SIZE` is a nonnegative integer, default **1024**; `0` disables caching.
Invalid values prevent startup. The cache has no TTL or persistence and starts
empty for each new application instance. Cache limits apply independently to each
process. Run one worker per container for consistent per-instance metrics and
cache accounting, then scale by adding containers. The optional local parser
loads models at startup and runs inference in a bounded worker thread. Its
performance must be assessed separately from the stub benchmarks below.

### Docker

Build and run with `.env` and automatic local E5 encoder mounting:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\docker-build-and-run.ps1
```

On Linux/macOS:

```bash
bash scripts/docker-build-and-run.sh
```

Copy `.env.example` to `.env` first if needed. Both scripts run in the foreground;
use `-Detach` / `--detach` for background mode, `-Port 8001` / `--port 8001` for a
different host port, and `-DryRun` / `--dry-run` to preview commands. Encoder
discovery checks the configured local path, `models/multilingual-e5-small`, then
the pinned Hugging Face cache snapshot. Override with `-EncoderPath` /
`--encoder-path`. Cached snapshot mounts preserve Hugging Face blob symlinks.
No model downloads are performed. See
[provisioning instructions](docs/optional-parser.md) if no encoder is available.
With `LOCAL_PARSER=tagger` the scripts also mount the provisioned tagger weights (`models/fine-tuned-tagger`,
read-only, at `/models/tagger`; override with `-TaggerPath` / `--tagger-path`) and refuse to start if they are
missing; run `python -m scripts.provision_tagger` first. See [the tagger parser](docs/tagger-parser.md).

Manual commands for the default stub mode:

```sh
docker build -t yad2-api .
docker run --rm -p 8000:8000 -e CACHE_MAX_SIZE=1024 yad2-api
```

The container runs as a non-root user, starts one Uvicorn worker, and includes a
health check against `/health`. The dataset editor is not part of the running API.

### Metrics and logs

`http_requests_total{route,method,status}` counts completed requests, including
validation errors and HTTP 500 responses. `http_request_duration_seconds` records
latency with a 150 ms bucket. `parse_requests_total{category}` counts successful
parses. Cache metrics are `parse_cache_hits_total`, `parse_cache_misses_total`,
`parse_cache_evictions_total`, and `parse_cache_entries`. With caching disabled,
each valid parse increments misses. Unknown paths use the bounded `unmatched`
route label. Scrapes include prior requests; the current scrape is counted after
its response is generated. No model token or cost metrics are emitted yet.

Example PromQL (use the same `instance` selector across expressions when needed):

```promql
# Parse error rate, including request validation failures
sum(rate(http_requests_total{route="/parse",status=~"[45].."}[5m]))
  / sum(rate(http_requests_total{route="/parse"}[5m]))

# Cache hit ratio
sum(rate(parse_cache_hits_total[5m]))
  / (sum(rate(parse_cache_hits_total[5m])) + sum(rate(parse_cache_misses_total[5m])))

# Parse p50; substitute 0.95 for p95
histogram_quantile(0.50,
  sum by (le) (rate(http_request_duration_seconds_bucket{route="/parse"}[5m])))
```

Request completion logs contain JSON with route, method, status, elapsed seconds,
category, and cache outcome. Query text and exception messages are excluded. Run
with `--no-access-log` to avoid additional unstructured Uvicorn access logs.

### Tests and throughput check

```sh
python -m pytest tests -q
python -m scripts.load_test --url http://127.0.0.1:8000 --output benchmark-results.json
```

Run the load check against a separate running server. Defaults: 20 offered
requests/second for 60 seconds **per workload**, after warm-up. The repeated-query
workload measures cache hits; the unique-query workload measures misses and, at
the default capacity, eviction. Each run uses a fresh query namespace. Both must
reach at least 12 successful requests/second, zero failures, and p95 latency at
most 150 ms for exit code 0. The report includes p50/p95, successful QPS, errors,
cache counter deltas, and client runtime information. Record server hardware,
runtime, worker count, and whether Docker was used alongside the report. Avoid
other traffic during measurement. `--rate` and `--duration` allow custom runs.

The script schedules requests independently of responses and includes scheduling
delay in latency. These results measure the stub infrastructure; they do not
establish performance for a future model-backed parser. Example-based tests,
validation failures, cache recency/eviction, concurrency, logging privacy, and
parser failure handling are covered alongside existing dataset tests.

Measured local results (one native Windows worker, Python 3.13.5, 60 seconds per
workload, 20 offered QPS):

| Workload | Successful QPS | Failures | p95 |
| --- | --- | --- | --- |
| Repeated query, cache hits | 20.0 | 0 | 18.0 ms |
| Unique queries, cache misses | 20.0 | 0 | 19.4 ms |

See [the benchmark report](docs/benchmark.json) for exact measurements and runtime
information, and [the metrics snapshot](docs/metrics.prom) for the resulting
Prometheus exposition. These are native-process measurements, not container
benchmarks or maximum-capacity estimates.

The Docker image was also built and smoke-tested: all three endpoints passed,
the runtime user was `appuser`, and Docker reported the container healthy.

## Goal
Design and implement a free-text search understanding service that converts a Hebrew query into structured Yad2 search parameters.

Your service should:

- Detect which marketplace vertical the query refers to:  
  - נדל״ן (Real Estate)  
  - רכב (Vehicles)  
  - יד_שנייה (Second-hand)  
- Extract and normalize the relevant structured filters for that vertical (e.g., city, rooms, price range, brand/model/specs).  
- Return a strict JSON response with the extracted parameters.  
- Be scalable, observable, cost-efficient, and secure (especially against prompt-injection).  

---

## Environment & Constraints
- **Language/Runtime:** Python  
- **Run mode:** Dockerized service running locally (provide Dockerfile)  
- **Input language:** Hebrew only, with tolerance to typos and slang  
- **LLM usage:** Any approach is allowed (rules, models, embeddings, hybrids).  
  - Must justify your choices and optimize for cost at Yad2 scale (tens of millions of searches/month).  
  - External embeddings/vector DBs are allowed.  
- **Secrets:** Use environment variables/config for API keys, never hardcode secrets.  

---

## API

### `POST /parse`
**Request body:**
```json
{ "q": "<free-text in Hebrew>" }
```

**Response body:**
```json
{
  "category": "נדל״ן | רכב | יד_שנייה",
  "params": { /* normalized filters per taxonomy */ },
  "confidence": 0.0,
  "notes": ["<optional normalization or assumption notes>"]
}
```

### `GET /health`
Returns simple health status.

### `GET /metrics`
Exposes observability metrics (Prometheus style is fine).

---

## Output Contract (Schema)
- **category** – enum: "נדל״ן", "רכב", "יד_שנייה"  
- **params** – object containing only allowed fields per taxonomy (`yad2_search_taxonomy_heavy.json`)  
  - Numeric ranges: `{ "min": <num>, "max": <num> }`  
  - Strings must be normalized (e.g., `"ירושלים" → "ירושליים"`)  
- **confidence** – float [0,1]  
- **notes** – optional array of strings with assumptions/normalizations  

Reject/flag unknown fields. Do not invent keys outside the taxonomy.  

---

## Non-Functional Requirements

### Scale & Performance
- Latency targets:  
  - p95 ≤ 600ms (model path)  
  - p95 ≤ 150ms (cache/rules only)  
- Throughput: ≥ 12 QPS per instance (~1M/day).  
- Caching: propose & implement (popular queries, partial-parse, normalization).  

### Cost Efficiency
- Track & expose token usage and cost/request.  
- Provide monthly cost estimate for **10M queries/month**.  
- Suggest cost reduction options (classifier + selective calls, embeddings+rules, prompt compression, caching).  

### Observability
Export metrics:
- Total requests / per category  
- Error rate  
- p50/p95 latency  
- Cache hit ratio  
- Token usage & cost/request  
- Model call success/failure rate  
- (Optional) Tracing with OpenTelemetry  

Use structured logs for parsing decisions & security events.  

### Security
- Fixed system prompts & allowlisted fields/categories.  
- Strict JSON Schema validation.  
- Input sanitization (strip emojis/control chars, normalize units).  
- Red-team tests for prompt-injection and abuse (long inputs, unicode tricks, slang, injection attempts).  

---

## Data & Taxonomy
Use the provided Hebrew taxonomy file: **`Yad2_search_taxonomy.json`**  
It includes fields for:  
- **נדל״ן:** transaction types, property types, city/rooms/size/floor ranges, amenities, typo mappings.  
- **רכב:** vehicle types, manufacturers → models, year/km/fuel/gearbox, safety features, typo mappings.  
- **יד_שנייה:** sectors, sub-categories, brand/spec fields, condition, price, location, typo mappings.  
- Global cleanup rules (units, typos, ranges).  

---

## Examples

### Example 1
**Query:**  
`דירת 3 חדרים בירושלים עד מליון שח`

**Response:**  
```json
{
  "category": "נדל״ן",
  "params": {
    "סוגי_נכס": ["דירה"],
    "מס׳_חדרים": 3,
    "עיר": "ירושלים",
    "מחיר": { "min": 0, "max": 1000000 }
  }
}
```

### Example 2
**Query:**  
`טויוטה קורולה 2018-2021 עד 70 אלף שח צבע לבן`

**Response:**  
```json
{
  "category": "רכב",
  "params": {
    "יצרן": "טויוטה",
    "דגם": "קורולה",
    "שנה": { "min": 2018, "max": 2021 },
    "מחיר": { "max": 70000 },
    "צבע": "לבן"
  }
}
```

### Example 3
**Query:**  
`אייפון 13 פרו 256 ג׳יגה כחול כמו חדש עד 2500`

**Response:**  
```json
{
  "category": "יד_שנייה",
  "params": {
    "סקטור": "אלקטרוניקה",
    "תת_קטגוריה": "טלפונים_סלולריים",
    "מותג": "אפל",
    "דגם": "iPhone 13 Pro",
    "נפח_אחסון": "256GB",
    "צבע": "כחול",
    "מצב": "כמו חדש",
    "מחיר": { "max": 2500 }
  }
}
```

---

## Deliverables
1. Code (Python) + Dockerfile.  
2. API docs (OpenAPI/Swagger or README).  
3. Design doc (README section): architecture, scaling plan, caching, cost model, rationale for chosen methods.  
4. Observability: `/metrics` endpoint with required metrics; provide snapshot/dashboard JSON.  
5. Security tests: unit/integration covering prompt-injection & invalid inputs.  
6. Example I/O: 5–10 realistic Hebrew examples with expected JSON.  

---

## Evaluation Criteria
- **Correctness & Robustness:** classification, extraction, normalization, schema adherence.  
- **Design Quality:** trade-offs for latency, cost, scale; caching strategy; modular architecture.  
- **Observability & Security:** useful metrics/logs/traces; prompt-injection defenses.  
- **Code Quality:** readability, tests, Dockerization, documentation.  
- **Pragmatism:** credible path to very low cost per query at Yad2 scale.  
