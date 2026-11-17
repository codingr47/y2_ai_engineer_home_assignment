# Setup

How to install, provision the models, run the API (natively or in Docker), and reproduce the evaluations. Run every
command from the repository root. The documented environment uses Python **3.13**; Docker uses Python 3.13 with
CPU-only PyTorch. Docker Desktop or Engine is required for container serving.

- [1. Create an isolated environment](#1-create-an-isolated-environment)
- [2. Get the models](#2-get-the-models)
- [3. Choose the serving profile](#3-choose-the-serving-profile)
- [4. Start the API](#4-start-the-api)
- [5. Tests, evaluations and benchmarks](#5-tests-evaluations-and-benchmarks)

## 1. Create an isolated environment

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
cp .env.example .env
```

Copy the example only on first setup, so an existing `.env` is kept. For CPU development, install the same PyTorch
version as the image before the remaining dependencies:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes the runtime requirements and test tooling. Scikit-learn is pinned to **1.7.1**, for
compatibility with the saved classifier artifacts.

Keep `.env` and model files out of Git. Never put API keys in source, image build arguments or evaluation reports.

## 2. Get the models

The large weights are **not in Git or in the Docker image**:
- the three fine-tuned tagger checkpoints, 370 MB each
- Frozen Fusion's E5 encoder, about 0.5 GB

A fresh checkout runs only in stub or OpenAI-only mode until they're supplied. There are two ways to get them.

### Option A: the artifact package (recommended)

The package (`yad2_artifacts_<commit>.zip`, about 1.6 GB) holds every model and runtime artifact at its
repository-relative path, plus a `MANIFEST.json` with each file's SHA-256. Unzip it **at the repository root**, then
verify:

```sh
python -m scripts.package_artifacts --verify-tree
```

Every file lands where the settings expect it (`models/fine-tuned-tagger/`, `models/multilingual-e5-small/`,
`src/artifacts/`, `src/extraction/artifacts/`). Bytes are kept exactly: both confidence gates hash these files and
refuse to load anything altered.

To build a package from a provisioned checkout:
```sh
python -m scripts.package_artifacts                                   # -> dist/yad2_artifacts_<commit>.zip
python -m scripts.package_artifacts --verify dist/yad2_artifacts_<commit>.zip
```

### Option B: provision from sources

Download the pinned E5 encoder once:

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-small', revision='614241f622f53c4eeff9890bdc4f31cfecc418b3', local_dir='models/multilingual-e5-small', allow_patterns=['*.json', '*.safetensors', '*.model', '1_Pooling/*'])"
```

The three trained ensemble checkpoints must match `src/artifacts/tagger/members.json`. If they're under
`training/training_bert/experiment1/runs/e6_lr5e-5_s{42,13,1337}/model.pt`:

```sh
python -m scripts.provision_tagger            # or --source <directory containing the three run folders>
```

The provisioner:
- checks each checkpoint's SHA-256
- copies them to `models/fine-tuned-tagger/`
- saves the DictaBERT configuration and tokenizer (this needs a cached copy or Hugging Face access)

It doesn't train or download the fine-tuned checkpoints.

**Safeguards:**
- **Serving downloads nothing.** Loading the gates checks code and artifact bindings, and loading the tagger checks
  member hashes.
- **Don't bypass these checks** or change `PARSE_CONFIDENCE_THRESHOLD` to make startup succeed.
- **Rebuilding:** changes to the model, extraction or gate inputs need the documented rebuild and validation workflow
  ([tagger setup and safeguards](docs/tagger-parser.md), [Frozen Fusion provisioning](docs/optional-parser.md)).

## 3. Choose the serving profile

For the full pipeline (the tagger, Frozen Fusion's opinion, and the Luna fallback), edit `.env`:

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

**How the settings combine:**
- `PARSER_MODE` overrides the legacy `EXTRACTOR_ENABLED` / `OPENAI_FALLBACK_ENABLED` selection flags.
- Keep `EXTRACTOR_ENABLED=true` for local and combined deployments launched with the Docker helpers: their mount
  discovery reads that flag independently.
- Use unquoted values in the Docker env file.
- In native Python, process environment variables override `.env`. The Docker helpers pass their env file to the
  container.
- Restart after configuration changes.

| Mode | Configuration | Assets / credentials |
| --- | --- | --- |
| Combined | `PARSER_MODE=combined`, `LOCAL_PARSER=tagger` | tagger weights, E5 encoder, shipped artifacts, OpenAI key and model |
| Local only | `PARSER_MODE=local`, `LOCAL_PARSER=tagger` | the same local assets; rejected queries return 503 without a fallback |
| Luna only | `PARSER_MODE=openai`, `EXTRACTOR_ENABLED=false` | OpenAI key and model; no local weights or encoder |
| Stub | `PARSER_MODE=stub`, `EXTRACTOR_ENABLED=false` | no models or key; returns a fixed fixture, for infrastructure checks |

**Notes on the profiles:**
- **The default:** the checked-in example uses **stub mode**.
- **The earlier local parser:** `LOCAL_PARSER=frozen_fusion` is still available.
- **The recorded Luna-only baseline** also sets `OPENAI_FEW_SHOT_COUNT=12`; the combined report used `0`.
- **The hybrid rule:** keep `TAGGER_HYBRID_VEHICLE_TYPE=false` to follow the training-label convention. Setting it to
  `true` also derives `סוגי_רכב: רכב היברידי/חשמלי` from a hybrid fuel type. That's a product decision, and against
  today's labels it costs about 2 points.

## 4. Start the API

Docker is the reproducible serving path. The helpers build the image, read `.env`, and mount the E5 encoder and tagger
weights read-only:

```powershell
.\scripts\docker-build-and-run.ps1
# Optional: -Detach -Port 8001; -DryRun previews the commands.
```

```bash
bash scripts/docker-build-and-run.sh
# Optional: --detach --port 8001; --dry-run previews the commands.
```

- **Asset locations:** override them with `-EncoderPath` / `-TaggerPath` or `--encoder-path` / `--tagger-path`.
- **What the helpers won't do:** download missing assets, or replace an existing container with the same name.
- **The image:** runs as a non-root user, with one Uvicorn worker and a `/health` health check.

For native development with the same `.env`:

```sh
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

**Checking it works:**
- **Liveness:** `GET http://127.0.0.1:8000/health`.
- **A parse:** try `POST /parse` in [Swagger UI](http://127.0.0.1:8000/docs). The OpenAPI contract is at `/openapi.json`.
- **The pipeline:** `/health` only shows the process is alive. To confirm which pipeline loaded, check the startup log
  and an actual parse.

PowerShell example:
```powershell
$body = @{ q = 'דירת 3 חדרים בירושלים עד מליון שח' } | ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:8000/parse -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

## 5. Tests, evaluations and benchmarks

Ordinary regression tests don't make paid calls. Model-dependent tests skip when the local weights aren't provisioned.

```sh
python -m pytest tests -q
```

**Data roles:**
- **Validation** is for development comparisons.
- **Test** is for a frozen configuration only. It has already been inspected, so treat it as regression evidence.

**End-to-end HTTP evaluation in Docker**, with the combined tagger pipeline. It builds the image, starts an isolated
container with the cache off, mounts the tagger and E5 encoder, and reports per request whether the local parser or
the LLM answered. Paid calls require explicit opt-in:

```sh
python -u -m scripts.evaluate_docker --mode combined --split test --allow-paid-calls
```

It reads the model and prompt settings from `.env`. Add `--limit 10` for a smoke run and `--split val` for development.

**Against an already running server:**
```sh
python -u -m scripts.evaluate_parser --mode api --url http://127.0.0.1:8000 --split val --limit 10 --allow-paid-calls
```

**An isolated Luna-only Docker smoke evaluation:**
```sh
python -u -m scripts.evaluate_docker --mode openai --split val --model gpt-6-luna --reasoning-effort low --few-shot-count 12 --guidelines on --limit 10 --allow-paid-calls
```

New runs use the current security prompt, so they won't exactly reproduce the historical Luna-only baseline. See the
[evaluation workflow](docs/openai-fallback.md) for artifacts, failure handling and cost accounting.

**Local-path throughput** needs no LLM and costs nothing. Start a local-only server (`PARSER_MODE=local`,
`LOCAL_PARSER=tagger`, `CACHE_MAX_SIZE=0`), then:

```sh
python -m scripts.benchmark_local --url http://127.0.0.1:8000 --rates 4 8 12 16 --duration 30 --output docs/benchmark_local.json
```

It sends unique validation queries open-loop and counts both 200 (answered locally) and 503 (rejected by the gate) as
completed local inferences.

**Security evaluation:** see [security behavior and tests](docs/prompt-injection-protection.md) for the Docker
red-team run.
