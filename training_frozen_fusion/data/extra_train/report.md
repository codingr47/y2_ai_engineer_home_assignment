# Extra training batch report (2026-11-16)

3,188 rows · 2,132 units · 0 gold · generated 3,250 · dropped 62

## Category

| category | rows | share |
|---|---|---|
| רכב | 987 | 31.0% |
| יד_שנייה | 987 | 31.0% |
| נדל״ן | 986 | 30.9% |
| null | 228 | 7.2% |

## Language

| language | rows | share |
|---|---|---|
| typo | 1,270 | 39.8% |
| standard | 961 | 30.1% |
| slang | 957 | 30.0% |

## Complexity

| complexity | rows | share |
|---|---|---|
| compound | 1,114 | 34.9% |
| simple | 980 | 30.7% |
| ambiguous | 866 | 27.2% |
| unsupported | 228 | 7.2% |

## Quality

| quality | rows | share |
|---|---|---|
| silver | 3,188 | 100.0% |

## label_seen_before

| label seen before | rows | share |
|---|---|---|
| False | 2,865 | 89.9% |
| True | 323 | 10.1% |

## Rephrasing

| is a rephrasing | rows | share |
|---|---|---|
| False | 2,132 | 66.9% |
| True | 1,056 | 33.1% |

## Edge case

| edge case | rows | share |
|---|---|---|
| none | 2,983 | 93.6% |
| multi_value | 29 | 0.9% |
| long_rambling | 26 | 0.8% |
| dont_care | 24 | 0.8% |
| distractor_numbers | 22 | 0.7% |
| abbreviations | 21 | 0.7% |
| unicode_noise | 19 | 0.6% |
| implicit_category | 18 | 0.6% |
| prompt_injection | 14 | 0.4% |
| very_short | 8 | 0.3% |
| negation | 7 | 0.2% |
| mixed_script | 6 | 0.2% |
| boundary | 5 | 0.2% |
| correction | 5 | 0.2% |
| fractional | 1 | 0.0% |

## Source

| source | rows | share |
|---|---|---|
| generator pipeline | 3,188 | 100.0% |

## Language × category

| category | typo | standard | slang |
|---|---|---|---|
| נדל״ן | 371 | 292 | 323 |
| רכב | 392 | 296 | 299 |
| יד_שנייה | 431 | 297 | 259 |
| null | 76 | 76 | 76 |

## Dropped

| reason | rows |
|---|---|
| blind extraction disagreed (not reviewed) | 62 |

## Leakage

Against the existing pool (every split, synthetic*.jsonl, annotations, the other batches, the parser repo's split1 and the earlier exports). Units inside the batch: 2,132 (1,056 with 2 rows, largest 2); merges: family 1,056; near-duplicate threshold: 4-gram Jaccard ≥ 0.6

| check: train batch vs existing | leaks |
|---|---|
| shared id | 0 |
| shared family | 0 |
| same normalized query | 0 |
| near-duplicate | 0 |
| contained query | 0 |
| identical label | 63 (allowed: tagged `label_seen_before`) |

**No query leakage: no train row matches an existing row.**
