# Y2 routing batch (2026-11-16)

**Purpose: calibrating and auditing the parser's confidence gate. Never train, fine-tune, early-stop or tune
hyperparameters on it.** Fit the logistic gate on 60%, choose its threshold on 20%, audit on the last 20%. Split
**by `unit`**, never by row: the two rows of a unit (a query and its rephrasing) must land in the same part.

Every row is new and unseen by the models trained on the existing splits:
- new ids (`rt_` from the generator pipeline, `rtgt_` written directly, outside the pipeline), new families and units;
- no row matches any existing row (all splits, every `synthetic*.jsonl`, annotations, the parser repo's
  `training/data/split1`, the earlier exports) by the split's leakage rules: the same normalized query,
  near-duplicate (character 4-gram Jaccard >= 0.6) or contained query. See the leakage table in report.md;
- identical labels are allowed (real traffic repeats intents) and tagged `metadata.label_seen_before`;
- nothing from the parser project's models or outputs was used to generate, filter or label it.

| | rows |
|---|---|
| total | 3,059 |
| units | 2,473 (at most 2 rows each) |
| gold (checked one by one by a person) | 236 (12 of them `rtgt_`) |
| human-written queries | 0 |
| silver (blind LLM extraction agreed) | 2,823 |

**Requirements not met yet: gold 236 of the 300 required; human-written queries 0 of 150.** 13 `rtgt_` rows wait for review and are not in this export.

## Who wrote the rtgt_ rows
162 `rtgt_` rows were written by Claude (an LLM, during the session that built this batch) directly, not
by the generator pipeline or a template, and were then checked by a person (gold) or bulk-approved after the blind
extraction agreed (silver). **They are LLM phrasing, not human phrasing**: they do not measure how the gate behaves
on real users' wording. 0 rows were written by a person. If the parser's LLM fallback is a Claude model,
keep in mind these rows share its authorship.


## Files
- `routing.jsonl`: one row per line, same schema as the dataset export's `train.jsonl`.
- `report.md` / `report.html`: distributions and the leakage audit.
- `manifest.json`: seed, generator / config version, ids, gold ids, sha256 of routing.jsonl, every existing file
  deduplicated against (with its sha256), drop counts and reasons.

## Schema
```
id         rt_XXXX (generator pipeline) | rtgt_XXXX (written directly; see "Who wrote the rtgt_ rows")
query      the Hebrew query as a user typed it (typos, slang and noise are intentional)
domain     נדל״ן | רכב | יד_שנייה | null (not a search)
expected   {"category": ..., "params": {field: value}}: the ORIGINAL field names (טווח_*, שנות_ייצור, ...)
metadata   domain, field_count, operator, language (standard/typo/slang), complexity,
           label_seen_before (true: the identical label occurs in the existing pool)
family     source id (a source and its rephrasing share it)
unit       leakage unit (family + near-duplicates + contained queries + identical non-empty labels in this batch)
quality    gold | silver
spans      {field: [{start, end, text, op, value, origin}]}: character offsets into `query`
```

## How it was made
The same generator and approval pipeline as the training data (planner in floor/gap mode, GLM / GPT / Gemini
rotation, blind LLM extraction that must agree, span verification, phrase budget), with one change: every
candidate is also checked against the whole existing pool by the leakage rules, and rejected (and regenerated)
when it matches. Mix targets: verticals about 31% each and 7% non-searches (near-miss negatives included);
language 40% typo / 30% standard / 30% slang; complexity and edge cases as in the training data.
