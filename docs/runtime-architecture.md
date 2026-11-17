# Runtime modules

The serving application is organized by responsibility. Implementations live in
packages; the few remaining root modules support existing launch commands and
untouched training callers.

| Package | Responsibility |
| --- | --- |
| `src.api` | FastAPI app factory, lifespan wiring, middleware, HTTP routes |
| `src.contracts` | Existing request, parse, search, and error models |
| `src.config` | Environment settings and repository-relative defaults |
| `src.parsing` | Parser protocol, stub, fallback exception and router |
| `src.frozen_fusion` | Frozen Fusion classifier, local parser, confidence gate |
| `src.llm` | OpenAI client/parser, catalog validation, prompts, reviewed examples, injection guard |
| `src.infrastructure` | Cache, Prometheus metrics, request-scoped usage |
| `src.extraction` | Existing field extraction package and extraction artifacts |
| `src.tagger` | Existing fine-tuned tagger inference, normalization and gate |

The API composes parsers. Parsers share contracts and use infrastructure; they
do not import API wiring. The tagger uses Frozen Fusion as its second opinion.
Configuration and contracts do not import parser implementations. Package
initializers perform no model/client loading. Only the application module builds
the existing ASGI application; lifespan still loads models and provider clients.

## Entrypoints and compatibility

Both commands serve the same application:

```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --no-access-log
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

Docker retains the second command. Runtime callers, tests, and operational scripts
use canonical package imports. `src.classifier`, `src.local_parser`,
`src.confidence`, and `src.settings` are legacy aliases to the actual implementation
modules, preserving module identity and training callers' monkey-patches. No other
old flat imports are supported. Training code and data were not edited.

All environment variables, artifact/weight locations, HTTP schemas/status codes,
prompts and reviewed examples are unchanged. The examples JSON moves alongside
its loader inside `src.llm`; taxonomy and model artifacts retain their locations.

## Gate-binding migration

Confidence artifacts hash runtime source. Moving imports changes those hashes
without changing fitted model behavior. The one-time audit command compares the
current code with main revision `b1a05cc15b2ac578d6237fb565f540c1133c0b56`, applying
an explicit module-name/resource-path allowlist before comparing executable ASTs.
Unapproved logic changes fail the audit. Both original source bindings must match
the starting artifacts, and original assets must remain unchanged.

```powershell
python -m scripts.migrate_runtime_bindings
python -m scripts.migrate_runtime_bindings --apply
```

The default is read-only. Applying changes only source/dependency bindings and audit
history, retaining estimators, feature schemas, thresholds, acceptance settings and
quality measurements. Frozen Fusion is updated before the tagger's nested binding
and gate-file digest. Full fitted state, fixed-vector scores and decisions are
compared. The command refuses to overwrite unrelated gate edits. It is specific
to this migration; do not use it to approve subsequent inference changes.

`runtime-module-migration.json` records old/new bindings and the source/state audit.
The existing `rebind_confidence_code` tool now discovers the relocated source files
and retains its stricter docstring-only rule for subsequent changes.

Runtime Python files use LF via `.gitattributes` because Frozen Fusion hashes raw
source bytes. Both gate loaders still reject mismatched bindings. Provisioned
encoder and ensemble weights are absent in this checkout: their original bindings
are preserved, and full model inference was not re-evaluated.

## Verification of this migration

- Offline suite: 316 passed, 7 skipped, and one pre-existing failure. The same
  failure occurred before any moves: `test_val_split_keeps_frozen_fusion_assignment`
  requires the absent frozen-training `val_split.json`. Training assets were not
  created or changed to mask it.
- OpenAPI, default settings, representative stub responses and all six prompt
  configurations matched the before-migration snapshot exactly.
- The final Docker image passed HTTP health, OpenAPI, three parse requests,
  invalid-request validation and metrics checks through the legacy entrypoint.
- Source AST audit, fitted-state/score parity, fail-closed binding checks,
  legacy module identity and monkey-patch compatibility passed. No paid API calls
  or training/recalibration were performed.
