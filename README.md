# Yad2 search understanding API

A Dockerized FastAPI service that converts Hebrew marketplace searches, including slang, typos, and numeric bounds, into structured filters for **נדל״ן**, **רכב**, and **יד_שנייה**.

The main serving pipeline combines a fine-tuned Hebrew tagger, Frozen Fusion as a second opinion, and a selective OpenAI fallback. It also supports local-only, LLM-only, and stub modes. See the [training journey](training/journey.html) for the experiment history and [runtime architecture](docs/runtime-architecture.md) for the code layout.

## How it works

```text
POST /parse -> request validation -> process-local LRU cache
                                    | cache miss
                  query cleanup -> 3-seed DictaBERT tagger
                                    |
                         spans + value normalization
                                    |
raw query -> Frozen Fusion -> agreement/confidence features -> tagger gate
                                                           |-- accepted -> local answer
                                                           `-- rejected -> Luna, low reasoning
                                                                         |
                                                        schema validation -> cache -> response
```

- **Tagger:** a three-seed DictaBERT-base ensemble predicts domain, field presence, spans, operators, and implied values. A deterministic normalizer and field-local extractor produce canonical values.
- **Frozen Fusion:** frozen multilingual E5 embeddings plus word/character TF-IDF and logistic-regression heads, with deterministic extraction. In the tagger pipeline its prediction and confidence inform the gate; it does not replace the tagger's answer directly.
- **Confidence gate:** scores whole-parse correctness and routes uncertain queries to the fallback. The shipped operating point is not a certified 95% accuracy guarantee; see [calibration details](docs/tagger-parser.md) and [gate provenance](docs/tagger_gate.json).
- **LLM fallback:** OpenAI Responses API, a fixed extraction/security prompt, a taxonomy catalog, and strict application-level output validation. The configured profile below uses `gpt-6-luna` with `low` reasoning. In LLM-only mode every uncached, allowed request takes this path.

Serving loads versioned artifacts from `src/artifacts/` and `src/extraction/artifacts/`, plus separately provisioned encoder/checkpoint files. It does **not** read training datasets or import training packages.

## What was achieved, and what remains

This is a working, evaluated API prototype. It demonstrates useful extraction and selective LLM routing on the repository's dataset; it does not establish production readiness or that all assignment targets were met.

| Area | Evidence and remaining limit |
| --- | --- |
| End-to-end parsing | The combined Docker HTTP run achieved **87.94% complete JSON accuracy** across 2,181 queries, including failures. It still produces incorrect filters; Hebrew typos, ambiguity, and distribution shift remain difficult. |
| Selective LLM use | That run answered **77.8% locally** and called Luna for **22.1%**. The estimated provider bill was lower than the recorded LLM-only run, but the runs have different prompt/product settings and do not isolate routing's effect. |
| Reliability | The shipped gate does **not** certify 95% local-answer accuracy. Its combined-run local answers were **90.39%** exact. A separate, stricter research calibration passed its audit bound at only 39.1% coverage; it is not the shipped operating point or a production guarantee. |
| Latency and capacity | Local answers had **109 ms p95** in the combined run. Overall HTTP p50 was **77 ms**, but p95 was **3.58 s**, above the assignment's 600 ms model-path target. **12 QPS with real model inference has not been demonstrated**; earlier stub benchmarks cannot establish it. |
| Security | Request/output validation, an LLM-path injection guard, a fixed security policy, and security tests are implemented. A small live suite passed 19/19 cases; this does not prove protection against unseen attacks. |
| Observability and cost | HTTP/cache/routing/provider/token metrics and sanitized logs are implemented. Dollar costs are offline estimates, not live cost metrics. Hosting cost, sustained-load behavior, and real production accuracy remain unmeasured. |

The test set has already been inspected during development and is now regression evidence, not an untouched estimate of performance on future traffic. Large local model assets must be provisioned separately; the default checkout starts in stub mode.

## Setup

Run commands from the repository root. Use Python **3.13** for the documented environment; Docker uses Python 3.13 with CPU-only PyTorch. Docker Desktop/Engine is required for container serving.

### 1. Create an isolated environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
cp .env.example .env
```

Copy the example only on first setup; preserve an existing `.env`. For CPU development, install the same PyTorch version as the image before the remaining dependencies:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes runtime requirements and test tooling. Scikit-learn is pinned to **1.7.1** for compatibility with the saved classifier artifacts. Keep `.env` and model files out of Git; never put API keys in source, image build arguments, or evaluation reports.

### 2. Choose the serving profile

For the tagger + Frozen Fusion opinion + Luna fallback, edit `.env`:

```dotenv
PARSER_MODE=combined
LOCAL_PARSER=tagger
# Also required by the Docker helper's model-mount discovery:
EXTRACTOR_ENABLED=true
EXTRACTOR_USE_TAGGER=true

OPENAI_API_KEY=<your-key>
OPENAI_MODEL=gpt-6-luna
OPENAI_REASONING_EFFORT=low
OPENAI_FEW_SHOT_COUNT=0
OPENAI_GUIDELINES_ENABLED=true
OPENAI_TIMEOUT_SECONDS=30
OPENAI_MAX_OUTPUT_TOKENS=4096

TAGGER_HYBRID_VEHICLE_TYPE=false
CACHE_MAX_SIZE=1024
```

`PARSER_MODE` overrides the legacy `EXTRACTOR_ENABLED` / `OPENAI_FALLBACK_ENABLED` selection flags. Keep `EXTRACTOR_ENABLED=true` for local/combined deployments launched with the Docker helpers: their mount discovery reads that flag independently. Use unquoted values in the Docker env file. Process environment variables override `.env` in native Python; the Docker helpers pass their env file to the container. Restart after configuration changes.

| Mode | Configuration | Assets / credentials |
| --- | --- | --- |
| Combined | `PARSER_MODE=combined`, `LOCAL_PARSER=tagger` | Tagger weights, E5 encoder, shipped artifacts, OpenAI key/model |
| Local only | `PARSER_MODE=local`, `LOCAL_PARSER=tagger` | Same local assets; rejected queries return 503 without a fallback |
| Luna only | `PARSER_MODE=openai`, `EXTRACTOR_ENABLED=false` | OpenAI key/model; no local weights or encoder required |
| Stub | `PARSER_MODE=stub`, `EXTRACTOR_ENABLED=false` | No models or key; returns a fixed fixture for infrastructure checks |

The checked-in example defaults to **stub mode**. `LOCAL_PARSER=frozen_fusion` remains available as the earlier local parser. For the recorded Luna-only baseline, additionally set `OPENAI_FEW_SHOT_COUNT=12`; the combined report used `0`.

Keep `TAGGER_HYBRID_VEHICLE_TYPE=false` to follow the training-label convention. Setting it to `true` additionally derives a hybrid/electric vehicle type from hybrid fuel; the combined HTTP report below used this product rule. That is a deliberate behavior difference, not a requirement for the tagger pipeline.

### 3. Provision local assets (combined/local modes)

The large weights and encoder are **not bundled in Git or the Docker image**. A fresh checkout cannot run the local pipeline until they are supplied.

Provision the pinned E5 encoder once:

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-small', revision='614241f622f53c4eeff9890bdc4f31cfecc418b3', local_dir='models/multilingual-e5-small', allow_patterns=['*.json', '*.safetensors', '*.model', '1_Pooling/*'])"
```

Obtain the three trained E6 member checkpoints matching `src/artifacts/tagger/members.json`. If they are already under `training/training_bert/experiment1/runs/e6_lr5e-5_s{42,13,1337}/model.pt`:

```sh
python -m scripts.provision_tagger
```

Otherwise pass `--source` with the directory containing those three run folders. The provisioner checks checkpoint SHA-256 hashes, copies them to `models/fine-tuned-tagger`, and saves the DictaBERT configuration/tokenizer (requiring a cached copy or Hugging Face access). It does not train or download the fine-tuned checkpoints.

Runtime serving performs no model downloads. Gate loading checks code/artifact bindings, and tagger loading verifies member hashes. Do not bypass these checks or change `PARSE_CONFIDENCE_THRESHOLD` to make startup succeed. Changes to model, extraction, or gate inputs require the documented rebuild and validation workflow: [tagger setup and safeguards](docs/tagger-parser.md), [Frozen Fusion provisioning](docs/optional-parser.md).

### 4. Start the API

Docker is the reproducible serving path. The helpers build the image, read `.env`, and mount the E5 encoder and tagger weights read-only:

```powershell
.\scripts\docker-build-and-run.ps1
# Optional: -Detach -Port 8001; -DryRun previews the commands.
```

```bash
bash scripts/docker-build-and-run.sh
# Optional: --detach --port 8001; --dry-run previews the commands.
```

Override asset locations with `-EncoderPath` / `-TaggerPath` or `--encoder-path` / `--tagger-path`. The helpers never download missing assets or replace an existing named container. The image runs as a non-root user with one Uvicorn worker and a `/health` health check.

For native development with the same `.env`:

```sh
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Check `GET http://127.0.0.1:8000/health`, then try `POST /parse` through [Swagger UI](http://127.0.0.1:8000/docs). The OpenAPI contract is at `/openapi.json`. `/health` is a liveness check; inspect startup logs and an actual parse to verify the selected pipeline.

## HTTP contract

| Endpoint | Purpose |
| --- | --- |
| `POST /parse` | Parse `{"q":"דירת 3 חדרים בירושלים עד מליון שח"}` |
| `GET /health` | `{"status":"ok"}` |
| `GET /metrics` | Prometheus metrics for this application process |

For example, in PowerShell:

```powershell
$body = @{ q = 'דירת 3 חדרים בירושלים עד מליון שח' } | ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:8000/parse -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Successful responses contain `category`, `params`, `confidence` (0–1), and `notes`. Categories are `נדל״ן`, `רכב`, `יד_שנייה`, or `null`; LLM-handled non-search queries return `category: null` and `params: {}`. Fields and canonical values follow the versioned runtime taxonomy. Numeric filters can be scalars or `{min}`, `{max}`, `{min,max}` objects. Vehicle output uses public `יצרן` / `דגם`, mapping to internal `יצרנים` / `דגמים`.

`q` must be the only request field: a nonblank string of at most 2,000 characters before trimming. Surrounding whitespace is trimmed. Local confidence comes from the correctness gate; LLM confidence is a **model self-rating**, identified as uncalibrated in `notes`.

| Status | Meaning |
| --- | --- |
| 400 | LLM input guard recognized a prompt-injection attempt (`prompt_injection_detected`) |
| 422 | Invalid request body |
| 503 | Local escalation without fallback, or provider timeout/unavailability/configuration error |
| 502 | Provider refusal, incomplete response, or invalid output |
| 500 | Unexpected internal failure; sanitized response |

The injection guard runs before provider calls on the **LLM path**; it is not a global pre-filter for locally accepted answers. The fixed security policy, separately JSON-encoded query, allowlisted taxonomy, and output validation provide additional layers. Heuristic detection and schema validity do not guarantee semantic correctness or resistance to every attack. See [security behavior and tests](docs/prompt-injection-protection.md).

## Evaluation reports

These are recorded **Docker HTTP evaluations** of all **2,181 test queries**, with the application response cache disabled. Open the standalone HTML reports directly in a browser; adjacent `summary.json` and `records.jsonl` contain configuration, hashes, and per-query evidence.

| Recorded configuration | JSON exact | HTTP p50 / p95 | LLM calls | Estimated LLM cost / 10M queries |
| --- | --- | --- | --- | --- |
| [Tagger + Frozen Fusion opinion + Luna fallback](eval/results/api_test_20261117T134548Z_0af454ba/REPORT.html) · `gpt-6-luna`, `low` | **87.94%** | **77 ms / 3.58 s** | **482 / 2,181 (22.1%)** | **$258** |
| [Luna only](eval/results/api_test_20261117T102939Z_e7c60732/REPORT.html) · `gpt-6-luna`, `low` | **89.36%** | **2.06 s / 3.29 s** | **2,181 attempted** | **$1,133** |

Important differences when reading these results:

- **Combined:** zero few-shot examples, extraction guidelines and the security guard enabled, `LOCAL_PARSER=tagger`, `TAGGER_HYBRID_VEHICLE_TYPE=true`, and one HTTP client. There were 1,697 local answers, 482 provider calls, and two security rejections. The local-answer accuracy was 90.39%; the fallback subset was harder, at 79.67% JSON exact. [Exact run metadata](eval/results/api_test_20261117T134548Z_0af454ba/summary.json).
- **Luna only:** twelve reviewed training examples, guidelines enabled, eight concurrent clients, and the earlier prompt before the mandatory security guard/policy. Two invalid provider responses returned 502. [Exact run metadata](eval/results/api_test_20261117T102939Z_e7c60732/summary.json).
- Both use the same test-data hash; failures count as incorrect. Different prompts, product rules, security behavior, and concurrency mean this is **not a controlled comparison**. In particular, the historical Luna-only score is not a measured baseline for today's security-enabled prompt, and the combined report differs from the recommended profile's default-off hybrid rule.
- Token costs use the reports' **17 November 2026** price tables and observed provider prompt caching. Projections assume the same traffic mix, routing, tokens, caching, and prices; they exclude local inference, hosting, and taxes. The disabled application LRU and the provider's prompt-token cache are separate mechanisms.

Also relevant: [Docker security evaluation](eval/results/security_20261117T125937Z_c1121a25/REPORT.html) (19/19 cases, Terra/low; a finite security check), [few-shot selection and error analysis](docs/few-shot-evaluation.md), and the [training journey](training/journey.html). Historical training gate figures such as 84.5% local coverage describe a different experiment configuration; use the combined HTTP run above for its actual routing rate.

### Run checks and new evaluations

Ordinary regression tests do not authorize paid evaluations:

```sh
python -m pytest tests -q
```

Model-dependent tests skip when local weights are unavailable. For development comparisons use validation data; reserve test for a frozen configuration and treat the already-inspected test set as regression evidence.

For an end-to-end check of the **tagger combined** configuration, start the API with the profile above and `CACHE_MAX_SIZE=0`, keep the evaluator's model/prompt settings aligned with the server, then explicitly opt in to paid calls:

```sh
python -u -m scripts.evaluate_parser --mode api --url http://127.0.0.1:8000 --split val --limit 10 --allow-paid-calls
```

This evaluates the running server; its model metadata is declared by the evaluator's environment rather than independently verified. This checkout's `evaluate_docker --mode combined` runner does not configure/mount the tagger, so use the running tagger API for that evaluation. For an isolated **Luna-only** Docker smoke evaluation:

```sh
python -u -m scripts.evaluate_docker --mode openai --split val --model gpt-6-luna --reasoning-effort low --few-shot-count 12 --guidelines on --limit 10 --allow-paid-calls
```

Remove `--limit` for a full validation run. New runs use the current security prompt and will not exactly reproduce the historical Luna-only baseline. See [evaluation workflow](docs/openai-fallback.md) for artifacts, failure handling, and cost accounting.

## Operating practices and limits

- **Cache:** copied successful responses in a thread-safe, process-local LRU keyed by the trimmed query; default capacity 1,024, `0` disables it. No TTL or persistence. Failed requests are never cached. Concurrent misses can make duplicate model calls; there is no request coalescing.
- **Scaling:** start with one worker per container so model memory, cache, and metrics have clear ownership. Scale containers based on measured uncached load and provider limits. The reported p95 includes LLM calls and does not meet the original 600 ms model-path target; it is not a 12-QPS capacity certification. The [older stub benchmark](docs/benchmark.json) measures infrastructure only.
- **Observability:** `/metrics` exposes HTTP counts/durations, successful parses by category, cache counters, routing reasons, local confidence, OpenAI outcomes/latency/tokens, and security rejections. Token kinds include input, output, cached input, and reasoning; the last two are subsets, not extra tokens to add again. Monetary cost is estimated in evaluation reports, not exported as a live dollar metric.
- **Logs:** structured request and decision records omit query text, provider exception messages, and credentials. Keep Uvicorn access logs disabled as shown. Evaluation reports intentionally include dataset queries and responses; handle them separately from operational logs.
- **Provider behavior:** one async client per app lifetime, 30-second default timeout, no automatic client retries, and a 4,096-token output limit including reasoning. Budget for failures and rate limits; unsupported model/effort combinations fail explicitly.
- **Deployment boundary:** the API has no built-in authentication or rate limiting. Put public deployments behind an authenticated, TLS-enabled gateway with request/concurrency limits. Preserve artifact checks and evaluate security, accuracy, latency, and cost after changing the model, prompt, decoding, product rules, or gate threshold.

## Repository guide

| Location | Contents |
| --- | --- |
| `src/api`, `src/contracts`, `src/config` | HTTP app, contracts, environment settings |
| `src/tagger`, `src/frozen_fusion`, `src/extraction`, `src/llm` | Local inference, extraction, routing, LLM and security layers |
| `src/infrastructure`, `src/artifacts` | Cache, metrics, usage accounting, versioned serving artifacts |
| `scripts`, `tests` | Provisioning, Docker helpers, evaluation tools, regression tests |
| `eval/dataset`, `eval/results` | Dataset tooling and saved evaluation evidence |
| `training/` | Both training histories and [journey.html](training/journey.html) |
| `docs/` | [Architecture](docs/runtime-architecture.md), [tagger](docs/tagger-parser.md), [fallback](docs/openai-fallback.md), [security](docs/prompt-injection-protection.md), and [next-data plan](docs/data_generation_plan.md) |
