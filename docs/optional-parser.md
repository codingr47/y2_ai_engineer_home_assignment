# Optional local parsing and confidence routing

The optional API pipeline uses frozen Exp3 v2 fusion classification (E5 + TF-IDF),
deterministic extraction, and an optional span tagger. Stanza morphology has been
removed from active code; historical reports retain its measurements.

## Configuration

Copy `.env.example` to `.env`. Process environment variables take precedence.
Settings are fixed at application creation; restart to apply changes.

| Variable | Default | Meaning |
| --- | --- | --- |
| `EXTRACTOR_ENABLED` | `false` | Enable local parsing and confidence routing; otherwise retain the stub. |
| `EXTRACTOR_USE_TAGGER` | `true` | Learned span fallback after deterministic matching fails. |
| `CLASSIFIER_ARTIFACT_PATH` | `src/artifacts/classifier.joblib` | Exported frozen Exp3 runtime bundle. |
| `E5_MODEL_PATH` | `models/multilingual-e5-small` | Provisioned local SentenceTransformer directory. |
| `CONFIDENCE_ARTIFACT_PATH` | `src/artifacts/confidence.joblib` | New audited confidence artifact; absent by default. |
| `PARSE_CONFIDENCE_THRESHOLD` | artifact threshold | Optional finite `[0,1]` override; changing it invalidates the audited operating-point claim. |

Boolean settings accept `true`, `false`, `1`, and `0`. Invalid settings prevent
application creation. Enabled mode loads and warms required assets at startup;
missing assets or mismatched confidence artifacts prevent startup. An absent
confidence artifact permits startup but disables local acceptance. `/health`
remains a liveness check, not a calibration-status assertion.

The committed runtime classifier can be rebuilt from the original training bundle:

```sh
python -m scripts.export_classifier
```

Export verifies probabilities and selected fields on all development queries.
Serving does not import training modules. Scikit-learn is pinned to the artifact's
training version, 1.7.1. Only load trusted joblib artifacts.

Provision the encoder once before startup (serving never downloads it):

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-small', revision='614241f622f53c4eeff9890bdc4f31cfecc418b3', local_dir='models/multilingual-e5-small', allow_patterns=['*.json', '*.safetensors', '*.model', '1_Pooling/*'])"
```

Set `EXTRACTOR_ENABLED=true`, then start Uvicorn normally. Docker installs CPU-only
Torch. Mount the provisioned encoder read-only:

For automatic encoder discovery and mounting, use
`powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\docker-build-and-run.ps1`
on Windows or `bash scripts/docker-build-and-run.sh` on Linux/macOS. Both scripts
read `.env`, build `yad2-api:local`, and run a container named `yad2-api`. Use
`-Detach` / `--detach` to run in the background; `-DryRun` / `--dry-run` previews
commands. They never replace an existing named container or download missing models.

To mount the encoder manually:

```sh
docker build -t yad2-api .
docker run --rm -p 8000:8000 --env-file .env --mount type=bind,source=<absolute-model-directory>,target=/models/e5,readonly -e E5_MODEL_PATH=/models/e5 yad2-api
```

Classifier/extractor artifacts are included in the image. Put the confidence
artifact under `src/artifacts/` before building, or mount it separately. Relative
paths resolve from the process working directory. `.env` and the encoder directory
are excluded from the Docker build context; `.env` is also git-ignored.

## HTTP behavior and fallback interface

On cache misses a bounded worker thread classifies category/fields, extracts and
resolves values, validates field names/types/enum values/bounds, and scores whole-JSON
confidence. Numeric values retain the v2 scalar or `{min}`, `{max}`, `{min,max}`
shape. Vehicle `יצרנים`/`דגמים` map to public `יצרן`/`דגם`.

Confidence combines domain probabilities, selected/rejected field probabilities,
decision margins, missing-value counts, and extraction sources/scores. Raw
extraction scores are ranking features, not correctness probabilities. Missing
predicted values remain visible to scoring even when omitted from candidate JSON.

Valid candidates are accepted at `confidence >= threshold`. Unsupported/non-search
categories, invalid output, and empty extraction always require fallback. Threshold
overrides cannot bypass an absent or unsuccessfully audited confidence artifact.
Successful local responses retain `{category, params, confidence, notes}`.

Without a fallback implementation, requests needing escalation return HTTP **503**:

```json
{"detail":{"code":"fallback_required","reason":"confidence_unavailable","confidence":null,"threshold":null}}
```

Other reason codes: `low_confidence`, `unsupported_category`, `invalid_output`,
`empty_extraction`. Scores/thresholds are numeric when available. No partial params
or query text are exposed. Unexpected inference/fallback failures retain generic
HTTP 500 responses. Only successful validated responses are cached.

No LLM provider is integrated. Connect one later through
`create_app(fallback_parser=...)`, implementing the existing asynchronous
`Parser.parse(query) -> ParseResponse` interface. It receives the original trimmed
query once. Its validated response replaces the local candidate. Explicit
`create_app(parser=...)` injection overrides environment-based parser selection.

Metrics add `parse_routing_total{destination,reason}` and `parse_confidence` on
cache misses. Decision logs contain destination, reason, score, and threshold,
without query text or unbounded labels. Token/cost metrics remain deferred.

## Separate calibration-data handoff

`--data` takes the generator's routing export directly (rows with `unit`, original
field names; converted to v2 at load time, `unit` used as the group), or rows such as:

```json
{"id":"routing-001","query":"דירת 4 חדרים עד 2 מיליון","domain":"נדל״ן","expected":{"params":{"סוגי_נכס":"דירה","מס׳_חדרים":4,"מחיר":{"max":2000000}}},"group_id":"query-family-001","language":"standard"}
```

Labels use the extractor's v2 taxonomy keys, including vehicle `יצרנים`/`דגמים`.
Non-search rows use `domain:null` and `expected:{"params":{}}`. `language` is an
optional slice label. Queries must be nonblank (surrounding whitespace is kept as
user noise). Related variants must share a `group_id`. Labels require independent human review.

```sh
python -m scripts.calibrate_confidence --data <routing-data.jsonl> --output src/artifacts/confidence.joblib --report docs/confidence_audit.json
```

The command rejects duplicate IDs, duplicate normalized queries, and overlap with
existing train/validation/test queries. It needs the training data tree for that
check, but never retrains the classifier/extractor. Existing experiment data must
not be reused as routing data.

Groups split reproducibly with seed 42: 60% confidence fitting, 20% threshold
selection, 20% untouched audit. Fitting uses standardized logistic regression
targeting complete category/params correctness and requires correct and incorrect
examples. At least five groups are structurally required; hundreds or thousands
of independently labeled queries are more useful for establishing reliability.

Choose maximum accepted coverage whose **one-sided 95% Wilson accuracy lower bound
is at least 96%** on selection (`--selection-target`; the margin offsets the
optimism of picking the threshold that only just passes). Audit the frozen
threshold against a lower bound of **at least 95%**. Audit results never tune the threshold. Reports include counts,
accuracy, coverage, fallback rate, lower bounds, category/language slices, quality/label-seen-before
slices, split IDs, and artifact hashes. Wilson bounds count accepted **groups**, not
rows (effective sample size = accepted groups at the row-level accuracy), because
a query and its rephrasing are not independent. Gold values outside the extraction
schema are kept and count as incorrect parses.

A failed selection/audit writes an acceptance-disabled artifact and exits **2**;
a passing audit exits **0**. Invalid input errors before publishing an artifact.
A disabled artifact can supply diagnostic scores but cannot enable acceptance.
Artifacts bind to classifier, encoder content, extraction assets, tagger choice,
inference code, and feature schema. Asset/code changes require recalibration.

**Current result (2026-11-16): acceptance stays disabled.** Run on
`training/data/fresh_routing_calibration` (3,059 rows, 2,473 units; synthetic and
LLM-written, no human-written queries), record in `docs/confidence_audit.json`:

- whole-parse exact accuracy on the new data is 68.8% (fit), against 81.1% on
  val_eval; typo queries are weakest;
- on selection the most confident 10% / 20% of queries are about 93% / 95% correct,
  a group-level lower bound of about 0.90, so no threshold reaches 0.96 and none
  is chosen; the audit accepted nothing and revealed nothing (it remains usable
  for one run after parser improvements).

No `confidence.joblib` is shipped: a disabled artifact behaves like an absent one,
and because artifacts bind to the inference code, keeping one would block startup
after any parser change. There is no numerical default threshold or 95% reliability
claim. Enabled extraction returns fallback-required 503s unless an injected
fallback handles those requests. Disabled mode remains the fixed stub.
