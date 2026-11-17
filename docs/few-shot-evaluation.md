# Reviewed few-shot reinforcement

The catalog, calibration, labels, and scoring are unchanged. The OpenAI parser
supports `OPENAI_FEW_SHOT_COUNT=0`, `6`, or `12`. The default remains zero until
a comparison establishes a preferred configuration; changing settings requires
an API restart. With guidelines disabled, zero-shot retains the original prompt byte-for-byte.

`OPENAI_GUIDELINES_ENABLED=true` adds the approved concise `IMPORTANT GUIDELINES`
section before the unchanged catalog. Disable it to reproduce the historical
prompts byte-for-byte. Docker evaluations accept `--guidelines on|off` and record
the flag and effective prompt hash. Existing comparison results were collected
before this section was added; their accuracy does not establish its effect.

`src/few_shot_examples.json` contains fixed, reviewed training queries, their
normalized expected category/params, source IDs, families, and review rationale.
The six-example set contrasts generic and explicit vehicle types, exact and
bounded years, calendar dates, and immediate entry. The twelve-example set adds
fuel alternatives, generic versus explicit condition, canonical city spelling,
neighborhood-only location, and selling versus searching.

Demonstrations appear in trusted instructions before the unchanged catalog.
They demonstrate extraction targets only; the model still supplies its own
uncalibrated confidence. Serving reads the committed artifact and imports no
training code. The artifact, selected IDs, catalog, and effective prompt are
fingerprinted in Docker evaluation reports.

## RCA evidence

The original test run contains 387 failures: 287 field-set differences, 70
value-only differences, 26 category errors, and four rejected provider outputs.
Its `RCA.html` and `RCA.json` retain each failed query, expected and actual output,
mechanical differences, overlapping root-cause hypotheses, review status, and
reinforcement decisions. Pattern identification is distinguished from confirmed
semantic causes. Suspected label conflicts are flagged and never silently fixed.
Rejected raw provider output was not saved, so those four cases remain unresolved.

Regenerate this evidence without model calls:

```powershell
python -m scripts.extraction_rca eval/results/api_test_20261117T085502Z_d07dc252
```

The test failures informed which behaviors to reinforce, so the test set is now
exposed. Test queries are not demonstrations. Source IDs, normalized queries,
and known families are checked against validation and test data before paid runs.
Training examples were silver labels; their inclusion follows explicit query
review rather than trusting their original quality tag.

## Docker comparison

```powershell
python -u -m scripts.evaluate_few_shots --allow-paid-calls --concurrency 8
```

This builds and launches the actual API for each configuration, uses HTTP
`POST /parse`, and evaluates the entire validation split for 0, 6, and 12 examples.
The same configured model, reasoning effort, concurrency, and disabled application
cache apply to every run. Validation comes from the existing frozen `val` split.
The source datasets are only read and are never edited.

The winning prompt has the highest validation whole-JSON accuracy, counting
failed HTTP requests as incorrect. Ties prefer fewer examples, then lower p95.
After selection, it runs once against `eval/dataset/data/test.jsonl` as an
**exposed regression comparison**, not an independent accuracy claim.

The comparison directory contains `REPORT.html` and `comparison.json`, with links
to each run's HTML, JSON, JSONL, and Markdown reports. It records corrected and
newly broken query IDs, category accuracy deltas, HTTP p95, failures, tokens,
estimated cost, provenance, and the selected example count. Completed validation
results are saved after every run, and HTTP row checkpoints are flushed throughout.
Production defaults and `.env` are not changed by the experiment.

To evaluate an individual configuration:

```powershell
python -u -m scripts.evaluate_docker --mode openai --split val --few-shot-count 6 --concurrency 8 --allow-paid-calls
```

The existing pytest Docker matrix also accepts `openai_few_shot_count` in each
`EVAL_CONFIGURATIONS` entry.
