# OpenAI fallback and evaluation

Enable local parsing and the fallback in `.env` or the process environment:

```dotenv
EXTRACTOR_ENABLED=true
OPENAI_FALLBACK_ENABLED=true
OPENAI_API_KEY=<your key>
OPENAI_MODEL=<your model ID>
OPENAI_REASONING_EFFORT=low
OPENAI_TIMEOUT_SECONDS=30
OPENAI_MAX_OUTPUT_TOKENS=4096
OPENAI_FEW_SHOT_COUNT=0
```

Install `requirements-dev.txt` and start the API as documented in the README.
Local model assets must be provisioned as before. Environment variables override
`.env`; configuration is fixed at app creation and changes require a restart.

`OPENAI_FALLBACK_ENABLED` defaults to false. Model and API key are required when
enabled; there is no hardcoded model selection. Reasoning effort is optional:
omit it for the model default, or set an effort supported by the selected model.
Values are forwarded without model-specific substitution. An incompatible model,
effort, or JSON mode combination fails explicitly. No temperature is set.
The default timeout is 30 seconds and the output limit is 4096 tokens, including
reasoning tokens. Client retries are disabled. One async client lives for the
application lifetime and is closed on shutdown.

`PARSER_MODE=stub|local|openai|combined` explicitly selects the serving pipeline
and overrides the legacy enable flags. `openai` serves OpenAI directly through
the normal API without requiring local assets. `combined` serves the local parser
with OpenAI escalation. If `PARSER_MODE` is omitted, legacy flag behavior is unchanged.

With `EXTRACTOR_ENABLED=false`, the original fixed stub remains active. An explicit
`create_app(parser=...)` overrides environment selection. An explicitly injected
`fallback_parser` takes precedence over the configured OpenAI provider.

## Parsing and response behavior

Accepted local queries avoid OpenAI. On escalation, a wrapper catches the existing
`FallbackRequired` exception and calls OpenAI. Local candidate generation, confidence
features, thresholds, source bindings, and calibration artifacts are unchanged.
No recalibration or artifact rebinding is performed.

The prompt adapts the synthetic generator's `EXTRACTOR_SYSTEM_TEMPLATE`, with a
catalog generated from `src/extraction/artifacts/taxonomy_v2.json`. Query text is
JSON-encoded in a separate user message; the fixed system prompt tells the model
to ignore instructions embedded in that query. The Responses API uses JSON-object
mode, `store=false`, and no tools. JSON mode ensures JSON syntax; application
validation checks the envelope, category-specific fields, conditional hierarchy,
enum values, numeric types and bounds. Validation does not establish that a
model's interpretation is semantically correct.

The HTTP response retains `category`, `params`, `confidence`, and `notes`. The
category now also permits null. Non-search, advice, and listing-posting queries
return HTTP 200 with `category: null` and `params: {}`. Consumers must handle null.
Fallback `confidence` is the model's self-rated probability of the entire parse
being correct, within `[0,1]`; `notes` explicitly identifies it as uncalibrated.
Local confidence retains its existing meaning. Vehicle output uses `יצרן`/`דגם`.
Successful validated responses, including non-search, use the existing LRU cache.

Without a configured provider, escalation retains the existing `fallback_required`
503 response. Provider errors use `{"detail":{"code":"..."}}`:

| Status | Codes |
| --- | --- |
| 503 | `provider_timeout`, `provider_unavailable`, `provider_configuration_error` |
| 502 | `provider_refusal`, `provider_incomplete`, `provider_invalid_output` |

Configuration errors include rejected model/effort combinations and authentication
errors. Provider error messages and query text are not returned or operationally
logged. Failed responses are never cached.

Prometheus exports `openai_calls_total{outcome}`, `openai_duration_seconds`, and
`openai_tokens_total{kind}`. Token kinds are `input`, `output`, `cached_input`, and
`reasoning`; cached input is a subset of input, and reasoning is a subset of output.
Evaluation cost estimates use a dated [OpenAI price lookup](openai-pricing.md),
including cached-input and cache-write rates. The lookup is separate from serving.

## Docker end-to-end evaluation

The model-comparison pytest suite builds the actual serving Docker image for
each configuration, starts a uniquely named container on an ephemeral localhost
port, waits for `/health`, and sends dataset queries as HTTP `POST /parse`
requests. It removes only its own container afterward, including on failure.
Images are retained and identified by tag and content ID in the report.

```powershell
python -u -m scripts.evaluate_docker --mode openai --split test --allow-paid-calls
python -u -m scripts.evaluate_docker --mode combined --split test --model <model-ID> --reasoning-effort low --allow-paid-calls
```

The default split is exactly `eval/dataset/data/test.jsonl`. Choose `--split val`
explicitly for model selection. `--limit 1` is a short smoke check. OpenAI API keys
are passed by environment variable name at container startup, never as build args,
image contents, command-line values, or report metadata. The selected model,
reasoning effort, timeout, output limit, and parser mode are runtime configuration;
the image is built from the current source and requirements. Cached Docker build
layers can be reused across configurations. `--reasoning-effort default` omits the
parameter even if `.env` configures an effort.
`--few-shot-count 0|6|12` selects reviewed training demonstrations. See the
[RCA and few-shot comparison workflow](few-shot-evaluation.md).

The default cache capacity is zero so results measure uncached API calls. Change
it with `--cache-size` for a cache workload; cache counter changes are reported.
`local` and `combined` require a provisioned encoder, mounted read-only, and use
the unchanged calibration artifacts. `openai` requires no encoder. API readiness
has a configurable `--startup-timeout` (default 180 seconds). Image build and
container startup time are excluded from request latency.
`--concurrency N` selects the number of simultaneous HTTP clients (default 1);
the report records it. Request timing excludes waiting for a client slot.
The pytest matrix accepts the same setting through `EVAL_CONCURRENCY`.

Reports record accuracy, client-observed HTTP p50/p95, successful-response p95,
HTTP status counts, server token and routing counter deltas, and the image/runtime
configuration. Failed requests count as incorrect and contribute to overall p95.
JSONL checkpoints are flushed after every response and retained if a run is
interrupted. Successful runs replace the checkpoint with the final reports.
Every completed run includes a standalone, searchable `REPORT.html` alongside
`REPORT.md`, `summary.json`, and `records.jsonl`. The command prints its HTML path.
Open the HTML file directly in a browser; it needs no server or external assets.
To add or refresh HTML for existing runs, use
`python -m scripts.evaluation_html eval/results`.
Reports also include estimated total cost, cost per query, and a 10M-query
projection, using reported token usage and the versioned price lookup.
New Docker runs include individual request cost and token usage in the HTML
query table and JSONL records. Evaluation-only usage headers provide accurate
attribution even with concurrent HTTP clients.

Use the pytest evaluation suite for a configuration matrix:

```powershell
$env:RUN_OPENAI_EVAL = '1'
$env:EVAL_MODE = 'openai'
$env:EVAL_SPLIT = 'test'
$env:EVAL_CONFIGURATIONS = '[{"openai_model":"<model-A>","openai_reasoning_effort":"low"},{"openai_model":"<model-B>","openai_reasoning_effort":null}]'
python -m pytest tests/eval/test_openai_evaluation.py -s
```

Each matrix entry builds and launches a container, then uses HTTP requests.
The opt-in flag authorizes paid calls. Omit it during ordinary regression testing.
For an API already running, use `python -m scripts.evaluate_parser --mode api
--url http://127.0.0.1:8000 --split test --allow-paid-calls`; ensure the server uses
the intended configuration. Its model metadata is declared by the client's `.env`
and is not independently verified as it is for Docker-managed runs.

## Direct parser diagnostics

The LLM gate includes mandatory [prompt-injection protection](prompt-injection-protection.md)
and a dedicated Docker HTTP security suite. Recognized attacks return 400 before
any OpenAI call.

The existing `RUN_EVAL=1` local candidate/gate test is unchanged. The additional
report runner measures actual parser responses without caching, using modes
`local`, `openai`, and `combined`. Local mode reports rejected queries as failures;
the old baseline still reports their candidate quality separately.

Use validation data for comparisons; reserve `--split test` for the final chosen
configuration. Validation defaults to the existing frozen training tree's `val`
split; test uses `eval/dataset/data/test.jsonl`. Labels are normalized to v2 at load
time. No labels, model artifacts, or thresholds are changed.

```powershell
python -m scripts.evaluate_parser --mode local --split val
python -m scripts.evaluate_parser --mode openai --split val --model <model-A> --reasoning-effort low --limit 50 --allow-paid-calls
python -m scripts.evaluate_parser --mode combined --split val --model <model-B> --reasoning-effort high --limit 50 --allow-paid-calls
```

For Docker-managed model comparisons, use the suite above. Direct diagnostic runs
exercise parser code only and do not measure HTTP latency. Omit `--limit` to
evaluate the full split. Direct calls are sequential so per-query token accounting
is unambiguous; this is not a throughput benchmark.

Each run writes a unique directory under `eval/results/` (or `--output`) containing:

- `summary.json`: configuration, dataset/prompt/taxonomy hashes, accuracy slices,
  failures, p50/p95 latency, token totals, routing and provider outcomes.
- `records.jsonl`: gold and predicted category/params, correctness, timing, usage,
  and sanitized failure code for each row.
- `REPORT.md`: readable accuracy, latency, token, slice, and routing summaries.
- `REPORT.html`: standalone, searchable results with cost estimates and per-query comparisons.

Reports are written even if individual provider calls fail. Failed rows count as
incorrect; the CLI exits nonzero and pytest fails after reports are saved. Existing
baseline reports are never overwritten. Non-search is correct only when category
is null and params are empty. Reports contain dataset queries and predictions for
error review; operational logs do not. No live comparison results are fabricated
by offline tests.
