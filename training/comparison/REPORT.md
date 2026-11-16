# Joint comparison: E5 vs TF-IDF vs fusion, 77 fields vs 70 slots

**Research question.** Does character/word-level TF-IDF carry information that the frozen E5 sentence embeddings
lose, and does combining the two improve complete slot detection?

**Short answer.**
- **Yes on both counts.** TF-IDF beats E5 by **+14.2 points** of slot exact-set accuracy on test (95% CI
  +11.9 to +16.5). Fusion adds another **+3.4** over TF-IDF (CI +1.8 to +5.0).
- **Fusion (Exp3) gives the best accuracy and the best confidence ranking** for an LLM fallback. **TF-IDF (Exp2)**
  is the low-latency option: 3.4 points behind, about 6× faster per query, and it needs no 493 MB encoder.

## Setup
| run | features | label space | selected on val_eval |
|---|---|---|---|
| Exp1 | frozen E5 (384-d) | 77 fields | C=64 (earlier) |
| Exp1-slots | frozen E5 | 70 slots | C=64 |
| Exp2 | TF-IDF word 1–2 + char_wb 2–5 (34,895 features) | 70 slots | C=16 |
| Exp3 | [w·E5, TF-IDF], both unit-norm | 70 slots | C=16, w=1 (joint 5×5 grid) |

- All four use the same one-vs-rest logistic heads, calibration and thresholds (`experiment_1/train.Heads`) and
  the same predicted-domain mask.
- **Data roles:** train fits models and feature extractors; **val_eval** selects C and w (its numbers are
  *development* metrics); **val_calib** is for calibration and thresholds only; **test** is scored once per
  experiment for the frozen configs.
- Exp1's test results had been inspected before these experiments. No choice here was tuned on test.
- Exp3 was picked from 25 configs vs 5 for the others, so its development score has slightly more selection
  optimism. The test gap confirms the ranking.
- **Slots:** the 77 fields contain **7** point/range twin pairs (`מחיר/טווח_מחיר` occurs in both נדל״ן and רכב).
  So the slot space has 70 labels, not the 71 in the plan. Twins never co-occur in a query, so merging is lossless
  for presence detection, but not for writing the output JSON (see the resolver section).
- The runner reproduces Exp1 exactly (val_eval exact 0.602, micro-F1 0.881) before any new run. Hashes of Exp1's
  model files and all data files are checked before and after.

## 1. Headline (`results/plots/headline.png`)
| model | split | domain acc | slot micro-F1 | slot macro-F1 | **slot exact-set** | strict exact-set¹ | domain ECE |
|---|---|---|---|---|---|---|---|
| Exp1 | val_eval (dev) | 0.961 | 0.893 | 0.859 | 0.631 | 0.602 | 0.015 |
| Exp1-slots | val_eval (dev) | 0.961 | 0.898 | 0.867 | 0.646 | 0.611 | 0.015 |
| Exp2 | val_eval (dev) | 0.982 | 0.944 | 0.903 | 0.767 | 0.731 | 0.011 |
| Exp3 | val_eval (dev) | **0.987** | **0.952** | **0.918** | **0.798** | **0.760** | 0.008 |
| Exp1 | test | 0.953 | 0.859 | 0.809 | 0.546 | 0.522 | 0.012 |
| Exp1-slots | test | 0.953 | 0.865 | 0.823 | 0.557 | 0.539 | 0.012 |
| Exp2 | test | 0.979 | 0.918 | 0.883 | 0.700 | 0.674 | 0.006 |
| Exp3 | test | **0.984** | **0.930** | **0.897** | **0.733** | **0.709** | 0.006 |

¹ Strict = the original 77 taxonomy names, using the deployable resolver (no operator yet, so each twin slot gets
its default name). Exp1 is mapped to slots for the slot columns.

The deterministic hierarchy rules (sub-category ⇒ sector, model ⇒ manufacturer) change slot exact-set by at most
+0.004. The models already predict parents almost always.

## 2. Family-level paired bootstrap (`results/plots/bootstrap_ci.png`)
Whole families (a source query plus its rephrasings) are resampled together: 2,000 replicates; 880 families in
val_eval, 1,664 in test. The value is the difference in slot exact-set accuracy (A − B).

| comparison | question | val_eval (dev) | test | test wins / losses |
|---|---|---|---|---|
| Exp1-slots − Exp1 | label space only | +0.015 [+0.003, +0.026] | **+0.011 [+0.004, +0.020]** | 47 / 22 |
| Exp2 − Exp1-slots | TF-IDF vs E5 | +0.122 [+0.090, +0.155] | **+0.142 [+0.119, +0.165]** | 490 / 180 |
| Exp3 − Exp1-slots | fusion vs E5 | +0.152 [+0.122, +0.183] | **+0.176 [+0.154, +0.197]** | 488 / 105 |
| Exp3 − Exp2 | fusion vs TF-IDF | +0.030 [+0.011, +0.049] | **+0.034 [+0.018, +0.050]** | 177 / 104 |

Every interval excludes 0 on both splits. The label-space change is small but real. The representation change is
the large effect. Fusion's gain over TF-IDF is modest but consistent.

## 3. What each representation contributes
**Per slot** (val_eval, dev; `results/plots/slot_delta.png`):
- TF-IDF vs E5: 44 slots improve, 18 regress. Gains are on surface cues E5 blurs: off-list brand `מותג` +0.35,
  `ריהוט` +0.35, `בעלות_מקרקעין` +0.23, `גודל_גלגל` +0.17, יד_שנייה `אזור` +0.16, `מצבי_עסקה` +0.15. Losses are on
  numeric/year concepts often written as words: `שנות_ייצור` −0.24, `ממ״ד` −0.14, `שנת_ייצור` −0.11,
  `מ״ר_בנוי⇄טווח_מ״ר` −0.11.
- Fusion vs TF-IDF: 31 improve, 18 regress (mostly small). Fusion recovers the numeric/year slots (`שנות_ייצור`
  +0.15, `ממ״ד` +0.14, יד_שנייה `מחיר` +0.11). Fusion vs E5: 46 improve, 11 regress.
- **Interpretation:** the two representations are complementary. TF-IDF carries lexical identity (brands,
  amenities, typo-tolerant character n-grams); E5 carries semantic paraphrase and spelled-out numbers.

**On the 100 cases from the Exp1 root-cause review** (cases whose full slot set is correct):

| primary cause (n) | Exp1 | Exp1-slots | Exp2 | Exp3 |
|---|---|---|---|---|
| typo / morphology (11) | 0 | 0 | 5 | 4 |
| point/range twin (10) | 7² | 7 | 8 | 9 |
| domain error (10) | 0 | 0 | 6 | 6 |
| prior-driven FP (7) | 0 | 0 | 5 | 4 |
| dilution in long queries (6) | 0 | 0 | 4 | 5 |
| spurious FP (6) | 0 | 0 | 5 | 5 |
| near-threshold (4) | 0 | 0 | 4 | 4 |
| entity knowledge (4) | 0 | 0 | 3 | 3 |
| embedding miss (4) | 0 | 0 | 2 | 2 |
| taxonomy overlap (4) | 0 | 0 | 2 | 3 |
| **controls (20)** | 20 | 20 | **17** | **19** |

² Exp1's twin-only errors count as correct at slot level by construction.

This supports hypothesis H1 from the Exp1 analysis: about half of the typo, dilution and embedding-miss cases are
fixed once lexical features are present. The control row is the price of TF-IDF alone: it breaks 3 of 20
previously correct cases, and fusion breaks 1.

## 4. Slices (test, frozen; `results/plots/slices_test.png`)
| slice | Exp1 | Exp1-slots | Exp2 | Exp3 |
|---|---|---|---|---|
| typo (713) | 0.488 | 0.504 | 0.634 | **0.676** |
| slang (717) | 0.565 | 0.570 | 0.734 | **0.759** |
| standard (751) | 0.583 | 0.597 | 0.730 | **0.763** |
| operator range (256) | 0.457 | 0.469 | 0.660 | **0.672** |
| operator min/max (477) | 0.430 | 0.463 | 0.589 | **0.652** |
| long queries, ≥13 tokens (693) | 0.489 | 0.505 | 0.649 | **0.700** |
| rephrasings (517) | 0.489 | 0.503 | 0.629 | **0.689** |
| sources (1,664) | 0.564 | 0.575 | 0.722 | **0.747** |
| **gold, hand-written/checked (717)** | 0.423 | 0.438 | 0.583 | **0.633** |
| silver (1,464) | 0.607 | 0.616 | 0.757 | **0.782** |

- Fusion leads in every test slice except non-searches (queries with no gold fields), where TF-IDF rejects
  better (0.915 vs 0.875).
- The largest fusion-over-TF-IDF gains are on long queries (+5.1), min/max operators (+6.3) and rephrasings
  (+6.0), where semantic context matters.
- **Typos remain the hardest register** (−8.7 vs standard).
- **Gold rows are 15 points harder than silver** for every model. This is a distribution shift from synthetic
  training data to hand-written queries, and the main risk for production accuracy.

## 5. Learning curves (`results/plots/learning_curves.png`, val_eval dev, 3 seeds)
| training rows | 250 | 500 | 1k | 2k | 4k | 6.2k |
|---|---|---|---|---|---|---|
| Exp1-slots | 0.202 | 0.316 | 0.416 | 0.524 | 0.599 | 0.646 |
| Exp2 | 0.198 | 0.318 | 0.463 | 0.597 | 0.709 | 0.767 |
| Exp3 | **0.225** | **0.361** | **0.498** | **0.648** | **0.742** | **0.798** |

- Fusion leads at every size.
- E5 and TF-IDF are level below 500 rows; TF-IDF pulls ahead from 1k.
- All three curves are still rising at 6.2k rows, so more (especially gold-like) data should help.

## 6. Point/range resolver (field-local; `taxonomy_rules.resolve(slot, local_span, operator)`)
| | val_eval | test | test gold | test silver |
|---|---|---|---|---|
| twin spans | 127 | 341 | 143 | 198 |
| span-local operator detector, 4-way | 0.976 | 0.950 | 0.979 | 0.929 |
| detector, bounded vs exact | 0.992 | 0.968 | 0.986 | 0.955 |
| gold name = default name (deployable) | 0.614 | **0.757** | **0.804** | 0.722 |
| gold name = taxonomy rule given detected op | 0.638 | 0.472 | 0.371 | 0.546 |
| gold name = taxonomy rule given **gold** op (oracle) | 0.630 | 0.463 | 0.371 | 0.530 |

- **Operator detection is not the bottleneck.** Given a field's own span, the cue lexicon gets bounded vs exact
  right 97–99% of the time. It sees only that field's span text, never the whole query; the cues were developed
  on train spans only.
- **The gold names do not follow the taxonomy rule.** Even with the true operator, the rule "bounded → `טווח_*`,
  exact → point" matches only 46% of test twin names and **37% of the hand-written gold rows**. Gold annotators
  mostly use the point name for bounds ("עד 2 מיליון" → `מחיר`).
- So strict field-name accuracy is capped by an unresolved labelling convention, not by the model. Applying the
  taxonomy-consistent rule *lowers* strict accuracy against current gold (test strict for Exp3: 0.709 with the
  default names, 0.685 with operators).
- **Decision needed (not a modelling task):** either the taxonomy's `טווח_*` filters are the intended output for
  bounded values (then gold should be relabelled), or the point fields carry bounds and the `טווח_*` names should
  be retired.
- **Mixed-operator queries are almost absent:** 0 in val_eval, 2 in test (both detected correctly). The
  field-local design is the right one ("דירת 4 חדרים עד 2 מיליון"), but this dataset cannot stress-test it. Targeted
  examples are needed.

## 7. Confidence features for LLM routing (`results/plots/risk_coverage_test.png`)
Per query, the records keep `domain_probs`, `slot_probs`, `predicted_slots`, `min_selected_probability`,
`max_rejected_probability` and `prediction_margin`, plus threshold-normalised versions. `slot_exact_match` is
stored as the evaluation target and is never an inference feature.

AUROC for predicting a fully correct slot set:

| score | Exp1 | Exp1-slots | Exp2 | Exp3 |
|---|---|---|---|---|
| domain confidence (test) | 0.556 | 0.561 | 0.580 | 0.570 |
| prediction margin, raw (test) | 0.735 | 0.758 | 0.763 | **0.782** |
| prediction margin, threshold-normalised (test) | 0.685 | 0.677 | 0.621 | 0.611 |
| min selected probability (test) | 0.676 | 0.700 | 0.689 | 0.734 |

| keep most-confident (Exp3 margin, test) | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---|---|---|---|---|---|
| slot exact-set on kept queries | 0.898 | 0.879 | 0.848 | 0.816 | 0.774 | 0.733 |

- **Domain confidence is close to chance** (AUROC 0.55–0.58) for "is the whole slot set right?". It cannot drive
  escalation alone.
- The raw selected-vs-rejected margin is the best single score, and Exp3's margin ranks best.
- The threshold-normalised margin ranks worse than the raw one. Per-slot thresholds distort comparability across
  slots.
- No router was built. A routing calibrator needs data not used for thresholds (val_calib predictions are
  in-sample for calibration), e.g. cross-fitted train predictions or a new held-out set.

## 8. Latency and size (`results/plots/latency.png`)
Measured on CPU (Intel 64 family 6 model 183, 16 torch threads); 200 val_eval queries one at a time, plus all of
val_eval in batches of 32. E5 model load (5.9 s) is excluded.

| model | batch-1 p50 | p95 | of which featurize | batch-32 per query | on disk |
|---|---|---|---|---|---|
| Exp1 / Exp1-slots | 21–24 ms | 28–31 ms | 18–20 ms (E5) | ~5.1 ms | 493 MB E5 + 0.3 MB heads |
| Exp2 | **3.5 ms** | **4.4 ms** | 0.6 ms | **0.21 ms** | 0.8 MB vectorizer + 20.7 MB heads |
| Exp3 | 20 ms | 24 ms | 17 ms (E5) | ~5.4 ms | 493 MB E5 + 21.7 MB |

E5 encoding dominates every E5-based model; classification is about 3 ms; post-processing is under 0.02 ms.
Run-to-run noise was about ±3 ms on the E5 rows. The TF-IDF heads are larger because the coefficient matrices
cover 35k features (they could be sparsified).

## 9. Conclusions and recommendations
1. **Use fusion (Exp3) as the field-presence model** when the ~20 ms per-query budget (or ~5 ms batched) is
   acceptable. It is the most accurate (test slot exact-set 0.733), has the best-calibrated domain head, and has
   the most useful margin for LLM escalation. **Keep Exp2 as the fast path** (3.5 ms, no encoder) if latency or
   footprint dominates; it costs 3.4 points.
2. **Keep the 70-slot space** for presence detection (+1.1, significant), and **resolve names in the taxonomy
   layer** from a field-local operator once the span/value extractor exists. First, settle the point vs `טווח_*`
   convention with the taxonomy owner: current gold contradicts the taxonomy for most bounded values.
3. **Close the gold-row gap** (−15 points vs silver): add hand-written or real queries to training, plus typo
   variants. All curves are still rising.
4. **Routing:** build the confidence calibrator on Exp3's raw margin plus slot/domain probabilities, fit on data
   not used for thresholds. Model `P(all slots correct | outputs)` now and `P(JSON correct | parser outputs)` once
   values are extracted.
5. **Data gaps to fill before trusting the operator path:** mixed-operator queries (≈0 today) and twin examples
   labelled under the agreed convention.

## Files
- `compare.py`: all numbers. It loads frozen models, scores val_eval and test, and checks artifact hashes.
  `plot_compare.py`: charts.
- `results/comparison.json`: full metrics, CIs, slices, per-slot deltas, resolver, confidence, latency.
  `results/plots/*.png`.
- `results/records_<split>.jsonl` (git-ignored, regenerable): per-query confidence features for all four models.
- Per-experiment: `training/experiment_1_slots/`, `experiment_2/`, `experiment_3/` (sweep, learning curve,
  final_test, README). Shared code: `training/shared/`.
