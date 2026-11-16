# Routing-batch error analysis and parser improvements (development)

**Why.** The first confidence-gate calibration (`docs/confidence_audit.json`) found the frozen parser only 68.8%
exact-JSON on the fresh routing batch, against 81.1% on val_eval. No threshold reached the 95% bound, so the gate
stays off. This folder finds where the gap comes from and fixes what can be fixed without new data.

## Data roles (changed for the routing batch)
| data | role here |
|---|---|
| routing **fit + selection** (2,448 rows, 1,978 units) | became **development** data. They had already been inspected during the first calibration |
| routing **audit** (611 rows, 495 units) | **sealed**: split off by the same seeded partition and never parsed, inspected or summarised. It remains valid for one final audit |
| split1 val_calib | thresholds and policies only (the proposal policy below) |
| split1 val_eval | development, as before |
| split1 test | not used |

Consequence: the gate's next fit and threshold selection need a **new** routing batch. The parser is now developed
against these rows, so its correctness on them is optimistic. The sealed audit can stay as the final check.

## Where the errors are (`analyze.py`, routing dev, frozen classifier)
First failing stage of the non-exact search rows (2,261 search rows):

| stage | baseline | after extractor fixes |
|---|---|---|
| classifier missed a gold field | 326 | 326 |
| wrong value | 134 | **98** |
| extractor found nothing for a selected gold field | 68 | 68 |
| extra field with a value | 55 | 53 |
| wrong domain | 12 | 12 |
| **JSON exact (search rows)** | **73.7%** | **75.4%** |
| value accuracy with gold fields (S1) | 94.7% | 95.4% |

- **The classifier's field recall is the main loss.** Given the gold field, the extractor nearly always finds the
  right value ("missed" fields are mostly cities, brands, colours, deal type, neighbourhoods).
- **Gold rows are not easier than silver** (70.5% vs 74–76%), so the gap is real parser error, not label noise.
  Some silver labels do omit fields (e.g. «מגרש» without `מצבי_עסקה: מגרשים`), but they are a minority.
- **By register** (after fixes): typo 66.0%, slang 81.0%, standard 82.4%. Rows whose label also occurs in the old
  pool score 84.5%, rows with new labels 72.3%: a coverage gap, not just noise.
- **Non-searches:** 24 of 187 are classified as a search, and 10 produce a payload. The gate must reject these.

## What was tried
1. **Extractor-proposed fields** (`proposals.py`): run the extractor on fields the classifier rejected and add
   confident hits. The policy was fitted on val_calib and evaluated on val_eval and routing dev. **Rejected:**
   - Precision is about 50–75% even just under the classifier threshold.
   - On val_calib it reverses (4 right vs 16 false near the threshold).
   - Most false proposals follow labelling conventions the classifier learned («יד שנייה» is not `מצב`, «אוטו»
     is not `סוגי_רכב: פרטי`, an on-list brand is `מותגים`, not `מותג`).
   - Net gain was +0.6 to +0.8 points, too small and unstable to ship.
2. **Typo augmentation of train** (`../experiment_6_typo_augment`): **no gain**, see its README.
3. **Extractor fixes** (shipped in `src/extraction/`). These are general rules with unit tests in
   `tests/test_extraction.py`:
   - **Booleans:** a negation inside the matched surface («בלי חיות» is itself a train surface), glued forms
     («בלימיזוג») and the typo «בליי».
   - **Ranges:** a coherent two-value range (min < max on one scale) outranks the halves it contains
     («בין 145 ל-160 מ״ר», «שתיים עד שלוש בעלים»), while 4 .. 2,000,000 is never one range.
     «בין קומה 0 לקומה 5» is now read as one range.
   - **Spelled-out ranges** «בין אלף חמישים לאלף ארבע מאות חמישים» now split into two values.
   - **Operators:** «ולפחות», «1+», «מאלף …» (a number word after the מ prefix).
   - **Place names with number words** («באר שבע», «קריית שמונה») no longer produce numbers.
   - **Typo'd longer values** beat the exact shorter value they extend word by word («כום חדש» → כמו חדש,
     «היברידי נטעם» → היברידי נטען, never LE → XLE).
   - **Fuzzy matching** never treats a frequent query word («מוצא») as a typo of a value.

   On val_eval (`results/extractor_fixes_val_eval.json`; the committed experiment_4 summary is left as the record
   of the first version):
   - S1 values: 96.9% → **97.3%**
   - S1 rows: 92.2% → 93.3%
   - JSON exact: 81.1% → **82.0%**
   - 11 values and 10 rows fixed, none broken.

## Next (needs data)
- **More training data.** The Exp3 learning curve is still steep: field exact-set 0.648 / 0.742 / 0.798 at 2k /
  4k / 6.2k train rows. A new generator batch for training (new units, the routing mix of registers) is the most
  likely way to lift field recall.
- **A new routing batch** for the gate's fit and selection, plus the sealed audit for the final check.

## Run
```bash
python training_frozen_fusion/experiment_5_routing_errors/analyze.py      # per-row records + summary_dev.json
python training_frozen_fusion/experiment_5_routing_errors/proposals.py    # proposal policy study (~4 min)
```
`summary_dev_baseline.json` is the frozen parser before the extractor fixes; `summary_dev.json` is after them.
Per-row records and caches are git-ignored.
