# Y2 synthetic search-query dataset (2026-11-15)

Hebrew free-text search queries for Yad2 (real estate, vehicles, second-hand goods) with structured labels and
token-level evidence spans, split for training a BERT-style tagger + classifier.

| split | rows | gold |
|---|---|---|
| train | 6,171 | 0 |
| val | 2,042 | 0 |
| test | 2,181 | 717 |

Open **report.html** for the distribution of every dimension by split, label coverage and findings.

## Files
- `train.jsonl`, `val.jsonl`, `test.jsonl`: one example per line (schema below).
- `report.html` / `report.md`: split coverage report.
- `manifest.json`: seed, fractions, ids per split, gold ids, hashes of the inputs.

## Schema
```
id         syn_XXXX (synthetic) or gt_XXXX (hand-written ground truth)
query      the Hebrew query as a user typed it (typos, slang and noise are intentional)
domain     נדל״ן | רכב | יד_שנייה | null (not a search: a negative example)
expected   {"category": ..., "params": {field: value}}; value = string | number | true/false |
           {"min": a, "max": b} | [several accepted values]
metadata   dimension tags: domain, field_count, operator, language (standard/typo/slang), complexity
family     the intent a row belongs to (a source and its rephrasings share it; never split across files)
quality    gold = checked by a person or hand-written; silver = bulk-approved after a blind LLM extraction
           agreed with the labels
spans      {field: [{start, end, text, op, value, origin}]}: character offsets into `query`;
           op = exact | min | max | range (numbers), true | false (booleans), value (everything else);
           origin = explicit (the field's own words) or derived (a hierarchy level implied by the item word,
           e.g. «ספה» for sector and sub-category)
```

## Labeling rules worth knowing
- Numbers and dates are written in the span, never calculated (a normalizer maps «150 אלף», «דונם וחצי»,
  «מרץ 2027» to the value).
- Free-text values are canonical (city spellings, no «שכונת» prefix); a city is never inferred from a street or
  neighbourhood.
- A brand on a sub-category's list is `מותגים` (list spelling); off-list brands stay in `מותג`.
- A powertrain named in the query is `סוג_דלק`; «מגרש» as the item implies `מצבי_עסקה: מגרשים`.
- Selling / renting out, advice questions, site support and chit-chat are not searches (category null).
