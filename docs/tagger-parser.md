# Fine-tuned tagger parser (`LOCAL_PARSER=tagger`)

The second local parser. It's the result of `training/training_bert/experiment1–3` and is opted into with one setting.
Routing, caching, logging, metrics and the LLM fallback are the same as the Frozen Fusion parser
([optional-parser.md](optional-parser.md)): `TaggerParser` subclasses `LocalParser` and replaces only how a candidate
parse and its confidence are produced.

```
query ─► runtime cleanup ─► 3-seed DictaBERT ensemble ─► span normalizer + extractor fallback ─► payload
   └───► Frozen Fusion on the raw query (its parse + gate score) ─────────────────────────────────────┐
                                              gate features ─► tagger gate ─► answer locally │ LLM fallback
```

## Results (test, scored once; see training/training_bert/experiment2)

| | JSON exact | gate at the 0.95 target | LLM calls (33M searches a month) |
|---|---|---|---|
| Frozen Fusion parser | 75.4% | 50% local at 86.5% | ~16.5M |
| **tagger parser** | **85.1%** | **84.5% local at 92.9%** | **~5.1M** |

**The 0.95 target isn't certified.**
- The val_calib audit lower bound is 0.943.
- On the routing-batch audit (test-like) the same threshold measures 95.5% at 84.3% coverage, with lower bound
  0.936.
- Numbers and provenance are in [tagger_gate.json](tagger_gate.json).
- To be stricter, raise `PARSE_CONFIDENCE_THRESHOLD`, for example 0.906, which gives about 77% local at 94.5% on
  test.

## Setup

```sh
pip install -r requirements.txt                      # torch comes with sentence-transformers
python -m scripts.provision_tagger                   # weights -> models/fine-tuned-tagger (git-ignored), SHA-256 checked
# the Frozen Fusion assets are needed too (second opinion): models/multilingual-e5-small, see optional-parser.md
```

`.env`, local only (rejected queries return HTTP 503 unless a fallback is configured):
```
EXTRACTOR_ENABLED=true
LOCAL_PARSER=tagger
```
The full pipeline: the tagger answers what its gate accepts, and OpenAI ([openai-fallback.md](openai-fallback.md))
answers the rest. That's `PARSER_MODE=combined`, or `OPENAI_FALLBACK_ENABLED=true` with `EXTRACTOR_ENABLED=true`:
```
EXTRACTOR_ENABLED=true
LOCAL_PARSER=tagger
OPENAI_FALLBACK_ENABLED=true
OPENAI_API_KEY=...
OPENAI_MODEL=<model id>
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `LOCAL_PARSER` | `frozen_fusion` | `tagger` selects this parser when `EXTRACTOR_ENABLED=true`. |
| `TAGGER_MODEL_DIR` | `models/fine-tuned-tagger` | Provisioned member weights + `encoder/` (config and tokenizer). |
| `TAGGER_ARTIFACT_DIR` | `src/artifacts/tagger` | Versioned artifacts: vocab, members (SHA-256), lexicon, numeric / common words, cleaner vocabulary, taxonomy. |
| `TAGGER_GATE_PATH` | `src/artifacts/tagger/gate.joblib` | The confidence gate (`tagger-gate-1`). |
| `TAGGER_HYBRID_VEHICLE_TYPE` | `false` | Product rule: a hybrid powertrain also sets `סוגי_רכב: רכב היברידי/חשמלי`. Off because today's labels don't follow it; turn it on together with the labelling-rule change. |
| `PARSE_CONFIDENCE_THRESHOLD` | artifact threshold | Shared override, as for Frozen Fusion. |

### Docker
```powershell
python -m scripts.provision_tagger                                     # once, on the host
.\scripts\docker-build-and-run.ps1                                     # or: bash scripts/docker-build-and-run.sh
```
- **What the run scripts read from `.env`:** `EXTRACTOR_ENABLED=true` and `LOCAL_PARSER=tagger`.
- **What they mount, both read-only:**
  - the E5 encoder at `/models/e5` (Frozen Fusion, the second opinion)
  - `models/fine-tuned-tagger` at `/models/tagger`, with `TAGGER_MODEL_DIR=/models/tagger`
- **Options:** `-TaggerPath` / `--tagger-path` overrides the host directory. `-DryRun` / `--dry-run` prints the
  commands.
- **The image itself contains no weights,** only `src/` with its versioned artifacts.
- **The gate's bindings are version-proof.** Code and artifacts are hashed with LF line endings; the encoder by its
  architecture fields and tokenizer vocabulary. So the gate built on the host loads inside the image, which runs its
  own `transformers` install.
- **Verified:** in the container, the parses and confidences match the in-process parser to 4 decimals.

## Shipping the models
The weights are git-ignored (370 MB each). To hand everything to someone, for example a reviewer:
```sh
python -m scripts.package_artifacts            # -> dist/yad2_artifacts_<commit>.zip (~1.6 GB)
python -m scripts.package_artifacts --verify dist/yad2_artifacts_<commit>.zip
```
- **What's in it, at repo-relative paths:** the 3 tagger members, the encoder files, the E5 encoder,
  `src/artifacts/` and `src/extraction/artifacts/`. Plus a `MANIFEST.json` with each file's SHA-256 and the source
  commit.
- **On the receiving side:** unzip at the repository root, then run
  `python -m scripts.package_artifacts --verify-tree`. Every file lands where the settings expect it.
- **Bytes are kept exactly.** Both gates hash these files and refuse to load anything altered.

## Safeguards
- **Every member checkpoint is checked** against `members.json` (SHA-256) before loading.
- **The gate refuses to load** if any binding changed:
  - the tagger artifacts
  - the encoder fingerprint (config + tokenizer vocabulary, not file bytes, so re-provisioning with another
    transformers version still matches)
  - the runtime code (`src/tagger`, `src/extraction`)
  - the feature schema
  - Frozen Fusion's bindings and gate file

  After changing any of these, rebuild with `python -m scripts.build_tagger_gate`.
- **The runtime never reads training data.** `src/tagger` imports no `training_*` module (a test checks this). The
  training code imports the runtime modules, so features, decoding and cleanup can't drift between training and
  serving.
- **Parity:** on 200 val_calib queries, the runtime matched the offline experiment payloads on 200/200 and the route
  on 199/200. The difference comes from CPU fp32 at runtime against bf16 on the GPU for most offline rows.

## Cost and latency
- **Latency:** about 77 ms per query on a laptop CPU for the whole chain (3 base-size encoders + Frozen Fusion).
  Startup takes about 9 s, mostly hashing and warming.
- **Memory:** about 1.5 GB of RAM for the three fp32 members. On a GPU, about 1.1 GB VRAM in bf16 (autocast is used
  automatically when CUDA is available).

## Rebuilding
```sh
python -m scripts.provision_tagger       # if weights are missing
python -m scripts.build_tagger_gate      # copy decoding artifacts, refit the gate exactly as experiment 2, bind it
pytest tests/api/test_tagger_parser.py   # end-to-end tests run when the weights are provisioned
```
Retraining (new data, see [data_generation_plan.md](data_generation_plan.md)):
- produces new member checkpoints and a new `members.json`
- refits the gate on the calibration batch
- needs a fresh test check
