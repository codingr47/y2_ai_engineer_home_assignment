# Experiment 1 – field classifier error analysis

**Scope.** The saved C=64 model (`artifacts/models/heads.joblib`) is used exactly as trained and calibrated: no
refit, no threshold change, no relabelling, no split change. Predictions come from the deployed path (`full`:
calibrated probabilities, tuned thresholds, predicted-domain mask) on **val_eval** (1,019 rows, the
model-validation half of val). The **test split is never loaded**. Train is read only for support counts, span
statistics and a kNN probe. The SHA-256 of `heads.joblib` and `thresholds.json` is checked before and after every
run (`94fbb8460a69794a`, `51aba1253c422a32`), and both are unchanged.

Baseline on val_eval: exact field-set accuracy **0.602**, micro-F1 **0.881**, **349 FN / 279 FP** field
decisions, 40 domain errors.

Every finding below is tagged **[M]** (measured on val_eval / train) or **[H]** (hypothesis, from reading the 100
sampled cases). Sample counts come from a *stratified* sample, so they show direction, not population rates.

---

## 1. Measured findings

### M1 – Domain errors are a small, separable share
- 40/1019 rows (3.9%) get the wrong domain. They carry **76 of 349 FN (22%)**, 28 of 279 FP, and 38 of 406
  exact-set failures (9%).
- On **31 of those 76 FNs, the field head was right**: the unmasked score was above threshold, and the
  predicted-domain mask removed it.
- If the domain were always right, exact set would go 0.602 → **0.614** (upper bound).
- Errors mostly involve יד_שנייה (it is on one side of 28 of the 40). 19 of the 40 involve `none` (a search sent to
  `none`, or a non-search accepted).
- With the domain right, exact set is 0.624, and failures split into FN-only 162, FP-only 134, both 72. Field
  detection dominates.

### M2 – Most misses are not near the threshold
- Correct-domain FNs (n=273): median score **0.10** vs median threshold **0.375**. Only **14.3%** reach 0.75×
  threshold, 31.9% reach 0.5×, and **25.6% are below 0.1×**.
- False positives are mostly confident: **41.8% score ≥ 2× threshold**, and only 37.5% fall within 1.5×.
- Re-picking per-field thresholds *on val_eval itself* (optimistic upper bound, never applied) gives micro-F1
  0.881 → 0.902 and exact set 0.602 → 0.652. Thresholds matter, but most of that gain is fitted to the
  evaluation data.
- **Extreme thresholds misbehave.** The 8 fields with threshold ≥ 0.5 have precision 0.958 but recall **0.741**.
  The 21 fields at ≤ 0.2 have recall 0.964, precision 0.888. Examples:
  - `נדל״ן/ריהוט`, threshold 0.75: recall 0.39, precision 1.00.
  - `נדל״ן/מצבי_עסקה`, threshold 0.2: 26 FP, precision 0.65.

  These per-field thresholds were tuned on val_calib fields with about 10–30 positives.

### M3 – The embedding neighbourhood usually doesn't carry the missed field
- Among a query's 10 nearest train queries (cosine on the frozen E5 vectors), the median share carrying the field
  is **0.2 for FNs vs 0.8 for TPs**. For **27.5% of FNs, no neighbour carries the field at all**.
- The pooled sentence vector places these queries among examples without the attribute. Moving a threshold
  cannot fix that.

### M4 – Point/range twin fields: the label convention overlaps
- The taxonomy has six point/range pairs, e.g. מחיר / טווח_מחיר, קומה / טווח_קומה, ק״מ / טווח_ק״מ. Train span operators
  show the "point" field regularly carrying bounds:

  | point field | point-field span ops | range field | range-field span ops |
  |---|---|---|---|
  | נדל״ן/מחיר | exact 34, min 21, max 18, **range 28** | נדל״ן/טווח_מחיר | min 19, max 20, range 11 |
  | רכב/מחיר | exact 21, **range 19**, max 14, min 13 | רכב/טווח_מחיר | min 13, max 13, range 9 |
  | רכב/ק״מ | exact 24, range 15, max 12, min 11 | רכב/טווח_ק״מ | min 14, range 10, max 11 |

- Upper-bound phrasings ("עד…", "תקרה…") on car fields are labelled `מחיר` 14 times and `טווח_מחיר` 12 times.
- The six pairs cause **120 of 628 field errors (19%)**. The most common same-query confusions are all twin swaps:
  `טווח_ק״מ→ק״מ` 6, `טווח_מחיר→מחיר` 5, `טווח_קומה→קומה` 4, `טווח_חדרים→מס׳_חדרים` 4, `רכב/מחיר→טווח_מחיר` 3.
- Merging each twin pair at evaluation lifts exact set 0.602 → **0.631** and micro-F1 → 0.893 (the largest
  single what-if).
- Recall by span operator: range 0.78, min 0.73, max 0.74, vs 0.89 for plain values. Whether the span contains a
  digit makes no difference (0.864 vs 0.871). Numbers themselves are not the problem; the operator/twin
  ambiguity is.

### M5 – Other overlapping or conditional labels
- Location: in יד_שנייה, spans starting "באזור…" are labelled `אזור` 109 times and `עיר` 17 times. `אזור→עיר`
  is the second most common confusion (5). Location fields (עיר / אזור / שכונה / רחוב) account for 91 errors.
  Merging יד_שנייה אזור/עיר at evaluation: exact set → 0.610.
- In רכב, spans containing "פרטי" are labelled `סוגי_רכב` 50 times and `בעלות` 29 times.
- **Deterministic, not inconsistent:** `אחסון_GB` appears only on laptops (92) and `נפח_אחסון` only on phones
  (114). `מותגים` covers electronics and strollers, while `מותג` (off-list brands) covers furniture, bikes and
  guitars. The field name follows another field's value, which a validation rule can enforce.

### M6 – Sparse supervision explains less than expected
- The correlation between a field's train support and its recall is weak (Spearman 0.17; with F1, 0.23).
- The 7 fields with fewer than 50 train examples do have recall 0.53, vs 0.94 for fields with 400+. Rare fields
  suffer, but most errors sit on fields with 50–400 examples.

### M7 – Where exact-set accuracy drops
| slice | exact set | note |
|---|---|---|
| typo vs standard language | 0.557 vs 0.626 | typo rows have 0.45 FN/query vs 0.25 |
| operator = range | 0.409 | vs 0.686 for queries with no operator |
| 1 expected field | 0.587 | **0.41 FP/query**: false positives dominate single-field queries |
| 5 expected fields | 0.440 | 0.73 FN/query |
| variation (rephrasing) rows | 0.525 | vs 0.605 for gap_fill; `item_last` 0.43, `question` 0.44 |
| edge-case rows | 0.488 | small n (43) |
| generator: gemini / glm / gpt | 0.516 / 0.625 / 0.650 | **gemini FP/query 0.36 vs gpt 0.18** |
| complexity simple / compound / ambiguous | 0.574 / 0.592 / 0.595 | complexity tag barely matters |

---

## 2. Root-cause review of 100 sampled cases (hypotheses)

The sample (`root_cause_sample.jsonl`) has 40 correct-domain FN cases spread over the most-missed fields, 30
correct-domain FP cases, 10 domain errors and 20 correct controls (10 of them hard: compound or ambiguous with 3+
fields). One reviewer labelled each case with a primary cause plus secondary causes
(`root_cause_annotations.jsonl`, codes defined in `annotate_sample.py`).

Primary cause across the 80 error cases (any-mention count in brackets):

| cause | primary | any mention | typical case |
|---|---|---|---|
| typo / morphology / mixed script hides the cue | 11 | 19 | `וכסוף!` → צבע p=0.03; `חמשלי` → סוג_דלק p=0.02 |
| point/range twin | 10 | 12 | "שלא עבר 220K ק״מ": gold טווח_ק״מ, predicted ק״מ at p=0.95 |
| domain head wrong | 10 | 10 | 4 of 10 had a correct field head (e.g. "וואו קיה סראטו יש בכלל באתר" → none) |
| prior-driven FP (no textual evidence) | 7 | 8 | מצבי_עסקה 0.84 with no deal word; מותגים 0.73 with no brand |
| dilution in long, chatty queries | 6 | 12 | price cue inside a 40-word message |
| spurious FP | 6 | 7 | טעינה_מהירה 0.81 for "יד שלישית או רביעית" |
| near-threshold | 4 | 8 | סוג_דלק 0.286 vs 0.325 |
| entity knowledge (street, neighbourhood, transliteration) | 4 | 5 | "כפר עקב", "בי וי די סיל" |
| embedding miss (explicit cue, short query) | 4 | 4 | "קרוב לבתי ספר" p=0.003 next to "בלי מיזוג" |
| taxonomy overlap (אזור/עיר, יד/מספר_בעלים, אבזור/מערכות_בטיחות) | 4 | 9 | "באזור השרון": gold אזור, predicted עיר 0.90 |
| lexical collision | 3 | 4 | "ברחובות" → רחוב; "יד 2" → תיבת_הילוכים 0.96 (ידנית) |
| rule-implied field | 3 | 4 | "מגרש" ⇒ מצבי_עסקה=מגרשים by labelling rule, p=0.05 |
| label inconsistency | 2 | 5 | "באזור חיפה" gold אזור vs "באזור באר שבע" gold עיר |
| sparse, conditional, negation, threshold setting | 6 | 11 | בעלות=השכרה for cars; ריהוט threshold 0.75 |

The 20 controls are mostly short, explicit queries with frequent values. Typos survive when the entity is frequent
(`סמסנוג`, `hליכון`). One control is a prompt-injection edge case, handled correctly.

**H1 – Pooled-embedding limits** (TYPO_MORPH, DILUTION, EMBEDDING_MISS, LEXICAL_COLLISION, NEGATION, about 25
primary). One 384-d vector per query loses short, local cues: a colour adjective with a prefix, a negated
amenity, one attribute among five. This fits M3 (missed fields absent from the neighbourhood) and M2 (misses far
below threshold).

**H2 – Labelling and taxonomy conventions** (TWIN, TAXONOMY_OVERLAP, LABEL_INCONSISTENT, CONDITIONAL,
RULE_IMPLIED, about 21 primary). Some errors are the model reproducing an inconsistent target. Supported by M4
and M5.

**H3 – Co-occurrence priors cause confident FPs** (PRIOR_FP + SPURIOUS, 13 primary). Independent one-vs-rest heads
learn "real-estate queries usually have מצבי_עסקה". This fits M2 (FPs mostly at ≥ 2× threshold) and M7
(single-field queries have the highest FP rate).

**H4 – Threshold choice is a minority cause** (THR_NEAR + THR_SETTING: 5 primary, 11 mentions). Consistent with
M2.

**H5 – Generator style** (M7: gemini rows carry twice the FPs of gpt rows). Possibly that generator under-labels
attributes it mentions, but the sample is too small to confirm. Needs a targeted audit.

---

## 3. Recommendations (not implemented)

Ranked by expected impact relative to effort and evidence. "Upper bound" means a what-if re-scoring of the
current predictions (M1, M4, M5), not a measured model improvement.

| # | change | expected impact | complexity | evidence |
|---|---|---|---|---|
| 1 | **Fix the point/range twin convention.** Either collapse each pair into one field and let the value extractor's operator (exact/min/max/range) pick the output key, or adopt a strict rule (bounded → `טווח_*`, exact → point) and relabel train/val | high: up to +2.9 exact, +1.2 micro-F1 | low (label mapping + retrain) | **strong [M4]**: overlapping ops in train, top-5 confusions are twin swaps, 19% of errors |
| 2 | **Taxonomy validation layer** (the planned JSON-assembly step): conditional field names (laptop → אחסון_GB, phone → נפח_אחסון; מותג vs מותגים by list lookup), rule-implied fields (מגרש ⇒ מגרשים, item ⇒ sector/sub-category), mutually exclusive pairs | medium: removes the CONDITIONAL / RULE_IMPLIED errors and part of the twins; not measured as one number | low–medium (rules, no training) | **strong [M5]** that the mappings are deterministic; **[H2]** for size |
| 3 | **Token-level span tagger** (the dataset already has character spans and ops) instead of, or alongside, pooled-embedding field detection | high: targets the largest hypothesis group and supplies operators for #1 | high | **[M3]**, **[M2]**, **[H1]** (~25 sampled cases) |
| 4 | **Cheap lexical channel**: char n-gram TF-IDF logistic regression stacked with the E5 heads | medium: typo/morphology and lexical cues | low | **[H1]**; a quick test of H1 before investing in #3 |
| 5 | **Use field evidence when choosing the domain**: pick the domain that maximises domain prob × in-domain field evidence, or route low-confidence domains instead of hard-masking | small: up to +1.2 exact | low | **[M1]**: 31 of 76 domain-error FNs had a correct field head |
| 6 | **Stabilise per-field thresholds**: shrink toward the global threshold, require more calibration positives, or tune on out-of-fold train predictions plus val_calib (never val_eval or test) | small–medium: the +5.0 oracle is optimistic; realistic gain mainly on the 8 high-threshold fields and on מצבי_עסקה | low | **[M2]**: extreme-threshold bands, ריהוט 0.75 / recall 0.39 |
| 7 | **Labelling-guideline fixes for location and ownership**: when to use אזור vs עיר for "באזור <city>"; "פרטי" as סוגי_רכב vs בעלות. Relabel affected rows | small: up to +0.8 exact for אזור/עיר | low–medium | **[M5]**: 17/126 and 29/79 split labels |
| 8 | **Reduce prior-driven FPs**: label-dependency modelling (classifier chains / a joint multi-label head), or negative examples showing a domain without its typical fields | small–medium | medium | **[H3]**, **[M2]**, **[M7]** |
| 9 | **Data**: audit gemini-generated rows for under-labelling; add typo/morphology and long-query variants for mid-frequency fields; add examples for the 7 fields under 50 train examples | medium, uncertain | medium | **[M7]**, **[M6]** (weak support effect), **[H5]** |

Suggested order: run **#1 + #2** together (both change only targets or post-processing, low risk), then **#4**
as a one-day check of H1 before committing to **#3**. Evaluate on val_eval, keep thresholds tuned on val_calib
only, and score the test split once at the end.

---

## Artifacts (this folder)
| file | content |
|---|---|
| `analyze.py` | reproduces everything except the manual labels (`python training/training_frozen_fusion/experiment_1/experiment_analysis/analyze.py`) |
| `predictions_val_eval.jsonl` / `.csv` | per query: id, query, gold/predicted domain and fields, all 77 calibrated probabilities, FP/FN fields, language, complexity, operator, quality, generation batch, generator, rephrase source |
| `diagnostics.json` / `.md` | per-field P/R/F1/support/FN/FP, confusions, FN/FP vs threshold, domain vs field errors, sparse-supervision, span-op and lexical-cue tables, exact-set breakdowns, what-ifs |
| `root_cause_sample.jsonl` | the 100 stratified cases with scores, thresholds and kNN rates |
| `annotate_sample.py`, `root_cause_annotations.jsonl`, `root_cause_summary.json` | manual cause labels (hypotheses) and tallies |

Generation batch and generator come from the generator's notes tags in
`../y2_ai_synthetic_generator/data/synthetic.jsonl` (read-only). Quality tier is `silver` for every val_eval row:
all gold rows are in test.
