# Routing batch report (2026-11-16)

3,059 rows · 2,473 units · 236 gold · generated 3,510 · dropped 451

## Category

| category | rows | share |
|---|---|---|
| נדל״ן | 983 | 32.1% |
| יד_שנייה | 915 | 29.9% |
| רכב | 912 | 29.8% |
| null | 249 | 8.1% |

## Language

| language | rows | share |
|---|---|---|
| typo | 1,213 | 39.7% |
| standard | 931 | 30.4% |
| slang | 915 | 29.9% |

## Complexity

| complexity | rows | share |
|---|---|---|
| compound | 956 | 31.3% |
| simple | 929 | 30.4% |
| ambiguous | 923 | 30.2% |
| unsupported | 251 | 8.2% |

## Quality

| quality | rows | share |
|---|---|---|
| silver | 2,823 | 92.3% |
| gold | 236 | 7.7% |

## label_seen_before

| label seen before | rows | share |
|---|---|---|
| False | 2,097 | 68.6% |
| True | 962 | 31.4% |

## Rephrasing

| is a rephrasing | rows | share |
|---|---|---|
| False | 2,473 | 80.8% |
| True | 586 | 19.2% |

## Edge case

| edge case | rows | share |
|---|---|---|
| none | 2,882 | 94.2% |
| unicode_noise | 26 | 0.8% |
| long_rambling | 21 | 0.7% |
| distractor_numbers | 19 | 0.6% |
| abbreviations | 18 | 0.6% |
| very_short | 14 | 0.5% |
| dont_care | 14 | 0.5% |
| implicit_category | 13 | 0.4% |
| negation | 10 | 0.3% |
| correction | 9 | 0.3% |
| multi_value | 9 | 0.3% |
| prompt_injection | 9 | 0.3% |
| mixed_script | 7 | 0.2% |
| boundary | 6 | 0.2% |
| unit_conversion | 2 | 0.1% |

## Source

| source | rows | share |
|---|---|---|
| generator pipeline | 2,897 | 94.7% |
| rtgt_ (claude) | 162 | 5.3% |

## Language × category

| category | typo | standard | slang |
|---|---|---|---|
| נדל״ן | 373 | 299 | 311 |
| רכב | 355 | 277 | 280 |
| יד_שנייה | 404 | 270 | 241 |
| null | 81 | 85 | 83 |

## Dropped

| reason | rows |
|---|---|
| unit cap: identical label | 429 |
| rtgt_, not reviewed yet | 13 |
| reviewer: drop | 9 |

## Leakage

Against the existing pool (every split, synthetic*.jsonl, annotations, the parser repo's split1 and the earlier exports). Units inside the batch: 2,473 (586 with 2 rows, largest 2); merges: family 586; near-duplicate threshold: 4-gram Jaccard ≥ 0.6

| check: routing vs existing | leaks |
|---|---|
| shared id | 0 |
| shared family | 0 |
| same normalized query | 0 |
| near-duplicate | 0 |
| contained query | 0 |
| identical label | 543 (allowed: tagged `label_seen_before`) |

**No query leakage: no routing row matches an existing row.**
