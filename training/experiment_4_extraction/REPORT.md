# Field-local value extraction: first development evaluation

**Question.** With the classifier frozen, how good is value extraction on its own, how good is the real pipeline,
and where does the gap between "right fields" and "right JSON" come from: span detection, numeric
interpretation, enum normalization, or other extraction failures?

**Answer** (val_eval = development, 1,019 queries, label schema v2, frozen Exp3 v2 classifier, `hybrid` extractor):
- **Scenario 1, gold fields supplied: 96.9%** of 2,680 values are exactly right, and 92.2% of searches have
  *every* value right. Extraction alone is not the bottleneck.
- **Scenario 2, predicted fields supplied:** value precision 0.955, recall 0.927, F1 0.941.
- **Scenario 3, complete JSON: 81.1%** exact, versus **79.9%** exact field sets. On this split there is no large
  drop from field-set to JSON accuracy:
  - Of the 814 rows with the exact field set, **55 (5.4 points)** still get a wrong value. Span detection causes 18
    of these, numeric interpretation 20, enum normalization 11, other extraction 6.
  - In **67 rows** the classifier predicted a spurious field, the extractor found no value for it, and the field
    was dropped, so the JSON came out exact anyway. The extractor acts as a second check on classifier false
    positives.
- **Classification remains the main loss.** 125 rows fail at the field set and 13 at the domain, against 55 at
  the values.

The 74.5% field-set figure quoted earlier is the *test* split. This evaluation is development-only, and test
was not scored. On val_eval the classifier's field-set exact is 79.9%.

## 1. What was built — `src/extraction/` (reusable; no API wiring yet)
`FieldExtractor.extract(query, field)` is **field-local**: it finds the span and value for one field. Nothing
looks at the whole query to decide another field's operator.

| component | role |
|---|---|
| `numbers.py` | Hebrew number reader, with token readers ported from the generator's verifier. Commits to ordered values per span ("בין 3 ל-4 אלף" → 3000, 4000), handles corrections ("2001 סליחה 2010" → 2010), דונם, TB, two-digit years |
| `operators.py` | span-local operator detector (moved from `training/shared`, which re-exports it) |
| `model.py` | per-field model built from **train**: type; surface → value lexicon (span texts, taxonomy values, aliases); numeric cue words and value ranges; units; common words |
| `match.py` | deterministic span finders: lexicon (longest match, prefix-tolerant), fuzzy (one edit), numeric candidates (bound and unit words, ranges, scoring by learned cues + plausibility), dates (digits and spelled-out), booleans with negation, cue patterns for street / neighbourhood / brand, size enums (64GB) |
| `taxonomy.py` | value rules: sector from sub-category, manufacturer from model, model from sub-model |
| `tagger.py` | learned fallback: per-field token classifier on train gold spans; threshold tuned on **val_calib** |
| `morph.py` | optional Stanza Hebrew segmentation and lemmas (off by default) |
| `extractor.py` | `extract`, `extract_all` (two numeric fields can't claim the same number; location fields can't share a span), `to_json` |

Variants evaluated:
- `lexicon`: deterministic matchers only.
- `hybrid`: plus the tagger fallback (default).
- `hybrid+morph`: plus the Hebrew morphology toolkit.

**Data roles:**
- Lexicon, cues and tagger: train only.
- Tagger threshold: val_calib (the sweep was flat at 0.960, so the neutral 0.5 was kept).
- All figures below: val_eval.
- Test: not used.

I developed the matchers by looking at val_eval errors, so these are development numbers. As an independent
check, scenario 1 on val_calib, which was not used for those choices, gives 96.0% versus 96.8% on val_eval
(lexicon variant).

## 2. Scenario 1 — gold fields supplied (`results/plots/extraction_scenario1.png`)
| field type | values | accuracy |
|---|---|---|
| enum | 2,149 | 97.6% |
| numeric | 386 | 93.8% (operator right on 95.6%) |
| boolean | 74 | 100% |
| date | 19 | 100% |
| open text (street, neighbourhood, off-list brand) | 52 | 84.6% |
| **all** | **2,680** | **96.9%** |

- **Span agreement with the gold span:** 2,051 exact, 558 overlapping, 44 disjoint, 27 not found.
- **Failure categories (83 values):**

  | group | count | breakdown |
  |---|---|---|
  | span detection | 38 | no span 27, wrong span 11 |
  | numeric interpretation | 23 | operator 5, number 7, both 11 |
  | enum normalization | 15 | wrong canonical value 12, derived 2, list partial 1 |
  | other | 7 | free text |

- **By slice:** derived values (implied by an item word) 98.4% vs explicit 96.7%; typo queries 94.9% vs standard
  97.8%.
- **By component:**

  | component | values | accuracy |
  |---|---|---|
  | lexicon | 2,100 | 99.1% |
  | numeric | 389 | 93.8% |
  | fuzzy | 104 | 94.2% |
  | cue patterns | 35 | 82.9% |
  | dates | 19 | 100% |
  | tagger | 27 | **11.1%** |

- **Weakest fields:** יד_שנייה `שנת_ייצור` (69%), `שכונה` (72%, mostly unseen names), `ריהוט` (78%), `קרבה`
  (84%), `ארנונה_חודשית` (86%).

## 3. Scenario 2 — predicted fields supplied
For each predicted field the extractor ran on the frozen classifier's output:

| outcome | pairs |
|---|---|
| right value | 2,485 |
| wrong value | 73 |
| gold field the classifier missed | 122 (13 of them from domain errors) |
| spurious field with an extracted value | 44 |
| spurious field with no value, so dropped from the JSON | 89 |

Value-level precision 0.955, recall 0.927, F1 0.941. The wrong values break down like scenario 1: no span 20,
wrong span 10, numeric 23, enum 14, free text 6.

## 4. Scenario 3 — complete JSON (`results/plots/extraction_waterfall.png`, `results/plots/extraction_gap.png`)
| stage | rows | share |
|---|---|---|
| all queries | 1,019 | 100% |
| domain correct | 1,006 | 98.7% |
| field set exact | 814 | 79.9% |
| … and every value right | 759 | 74.5% |
| **JSON exact** (adds 67 rescued rows) | **826** | **81.1%** |

| counterfactual | JSON exact |
|---|---|
| gold domain + gold fields (scenario 1, searches) | 92.2% |
| perfect extraction (= domain and field set exact) | 79.9% |
| **actual pipeline** | **81.1%** |

**Gap from field set to JSON: 55 rows, 5.4 points.** By first failing value category:

| category | rows | group |
|---|---|---|
| no span | 12 | span detection |
| wrong span | 6 | span detection |
| operator + number | 9 | numeric interpretation |
| number parse | 6 | numeric interpretation |
| operator | 5 | numeric interpretation |
| enum normalization | 10 | enum normalization |
| list partial | 1 | enum normalization |
| free text | 6 | other extraction |

JSON exact by register: slang 86.2%, standard 84.5%, **typo 73.3%**.

**About the rescued rows.** Dropping a predicted field because no value was found helps here, since most
classifier false positives have no supporting text. It also means a true field whose value the extractor misses
silently disappears from the JSON. For routing, "classifier says present, extractor found nothing" is a useful
uncertainty signal and should be kept.

## 5. Variants
| variant | S1 values | S1 rows | S2 F1 | S3 JSON | runtime for val_eval |
|---|---|---|---|---|---|
| lexicon | 96.8% | 91.8% | 0.941 | 80.9% | 2.6 s |
| **hybrid** | **96.9%** | **92.2%** | **0.941** | **81.1%** | 2.7 s |
| hybrid+morph | 96.8% | 91.9% | 0.940 | 80.8% | 54.5 s |

- **The tagger helps only at the margin.** It fires on 27 values the matchers couldn't place, and only 11% of
  those are right. Its gain (+0.1 point) comes from the few it recovers. It is worth keeping as a fallback, but
  not trusting.
- **The Hebrew morphology toolkit doesn't pay off here.** Its segmentation helps a few prefixed forms
  ("ברחובות" → רחובות). Its over-segmentation ("וכסוף" → ו+כ+סוף) and lemmas add enum errors (12 → 17). It is
  20× slower. It stays an optional switch.

## 6. Limitations and next steps
1. **Open-vocabulary text** (neighbourhood and street names never seen in train) is the weakest type. A gazetteer
   of Israeli neighbourhoods and streets would help more than a better model.
2. **Typo queries** lose about 11 JSON points against standard ones. Fuzzy matching covers one edit; glued words
   ("מעליתבבניין", "סוזוקיקרוסאובר") and multi-letter noise are not handled.
3. **Mixed-operator queries** are nearly absent from the data, so the field-local operator design ("4 חדרים עד 2
   מיליון" → rooms exact, price max) is covered by unit tests rather than by dev numbers.
4. **Before scoring test once:** freeze the extractor, then run `evaluate_extraction.py --final-test`.
5. **Routing features to add from the extractor:** the source component, the match score, and "predicted field
   with no value".
6. **Side effect on an earlier report.** Moving and slightly improving the span-local operator detector (train
   bounded-vs-exact 0.966 → 0.970) means re-running `training/comparison/compare.py` would shift its resolver
   diagnostics marginally.

## How to run
```bash
pip install -r training/requirements.txt          # stanza only needed for --morph
python training/experiment_4_extraction/build.py               # lexicon/taxonomy (+ tagger) from train
python training/experiment_4_extraction/evaluate_extraction.py --tune-tagger --morph
python training/experiment_4_extraction/plot_results.py
python training/comparison/report_html.py training/experiment_4_extraction/REPORT.md
```
`results/summary_val_eval.json` holds every number above. Per-value, per-pair and per-row records
(`results/*.jsonl`) and `tagger.joblib` are git-ignored and regenerable.
