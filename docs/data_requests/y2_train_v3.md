# Data request: training batch `y2_train_v3` (for the generator project)

**From:** the parser project, branch `exp/tagger-v2`.
**Machine-readable spec:** [`y2_train_v3.json`](y2_train_v3.json): buckets, mixes, and per-field quotas computed from
the current training data.
**Role: training only.** Never use this batch for model selection, calibration or test.

## Why
The live system (fine-tuned tagger + gate + LLM fallback) scores 87.9% JSON exact on test. On validation, 9.0% of
queries are wrong. An error analysis on validation only (test untouched) shows:

| slice | error rate |
|---|---|
| typo queries | **11.6%** (standard 8.2%, slang 6.0%) |
| real estate | **10.9%** (second-hand 9.1%, vehicles 8.1%) |
| "ambiguous" queries | **11.7%** |

**Rare fields fail 3–4× more often than common ones.** Neighbourhood is wrong 33% of the time, off-list brand 27%,
year of manufacture 21%, electric range 20%. 19 fields have fewer than 200 training examples.

**Most errors are things the model has never seen:**
- typo'd or rare place names
- brand spellings
- the many ways people name car models
- values stated without their keyword
- unusual number phrasing

## Deliverable
| item | value |
|---|---|
| export | `y2_train_v3_<date>/`: `train.jsonl`, `manifest.json`, `README.md`, `report.md`, same schema as `y2_dataset_*` |
| rows | **~9,300** (≈ $39 at ~$4.2 per 1k) |
| ids | a new prefix, e.g. `v3_…`, unique across all exports |
| schema | `id, query, domain, expected{category, params}, metadata{domain, field_count, operator, language, complexity}, family, unit, quality, spans` |
| labels | v2 schema (`normalize.py` canonicalized), exactly as the existing exports |
| spans | **required** for every field the query states, with exact character offsets; `origin` explicit / derived as today |
| quality | silver (bulk-approved after the blind extraction agreed) is fine; spot-check a random **2%** by hand and record the verdicts |

## Overall mix
- **Language register:** typo **45%**, slang 30%, standard 25%.
  - Today's train is 38% typo, and typo is the weakest slice.
- **Domain:** real estate **38%**, vehicles 30%, second-hand 27%, non-search 5%.
- **Field count:** 80% multiple, 15% single, 5% zero.
- **Typo kinds,** mixed, about one or two per typo query:
  - a glued word, or a glued one-letter prefix («בפתחתקווה», «לאמעלית»)
  - final-letter misuse (ם/מ, ן/נ, ף/פ, ך/כ, ץ/צ: «אלפ», «חדרימ»)
  - a Latin letter or digit inside a Hebrew word («חנyה», «מטב1ח»)
  - punctuation inside a word («מר.פסת»)
  - doubled or dropped letters
  - keyboard-adjacent swaps
  - phonetic misspellings of place and brand names («פתח תיקווה», «סאמסונג»)

## Buckets
The examples are invented to show each pattern. **Don't copy them;** vary the wording.

| bucket | rows | what | invented examples |
|---|---|---|---|
| **B1 places** | 1,500 (real estate 1,000, second-hand 500) | `עיר` / `שכונה` / `רחוב` / `אזור` across the **whole gazetteer**, not just the frequent cities. Each place in 2–3 realistic forms: typo'd, glued to a prefix, a common alternative spelling. Rare towns and neighbourhoods first. | «דירה בקרית מוצקן», «בשכונתנווה שאנן», «ברחוב הרצל ברחבות», «באיזורהשרון» |
| **B2 brands** | 1,000 (second-hand) | many brands in Hebrew and Latin spellings and typos. Use `מותגים` when the brand is on the sub-category's list, `מותג` when it isn't (existing rule). | «מקרר של סאמסונג», «אוזניות JBL», «אופני ג'ייאנט», «שעון גרמין» |
| **B3 car models** | 1,000 (vehicles) | how people name models and sub-models: spelt-out numbers («סדרה שלוש»), maker-less short forms («איוניק 5»), dotted abbreviations («מ.ב»), Hebrew/Latin mixes («קיה ספורטאג'»), sub-model trims | «ב.מ.וו סדרה שלוש», «מאזדה שלוש», «קורולה הייבריד» |
| **B4 meaning without the keyword** | 1,500 (vehicles 600, real estate 600, second-hand 300) | categorical values stated without their keyword, plus negations: fuel, ownership, deal type, condition, furniture, nearby places, safety systems, equipment | «שנוסע על חשמל ודלק» → `סוג_דלק: היברידי`; «לא מליסינג ולא מחברה» → `בעלות: פרטי`; «ריקה לגמרי, אפס ריהוט» → `ריהוט: ללא`; «עשר דקות הליכה מהים» → `קרבה` |
| **B5 numbers** | 1,500 (500 per domain) | unusual or misspelt bound words («אד», «לכל היותר», «לא פחות מ»); self-corrections («חמש מאות, בעצם ארבע מאות»); before / after / since phrasing («מלפני 2010» → `{max}`, «מ-2018 ואילך» → `{min}`); year ranges for items; electric range; storage in words and units («טרה וחצי», «256 ג'יגה»); prices in words | «טווח נסיעה מעל שלוש מאות וחמישים קמ», «מחשב עם חצי טרה», «מעל שלושת אלפים לחודש» |
| **B6 rare fields** | 2,300 | top-up so every field reaches its `min_new_rows` in the JSON. Rows from B1–B5 that contain a field count towards its quota. | — |
| **B7 boundaries** | 500 (non-search 250, the rest searches) | short noisy fragments that are still searches («3 חד ברמג»), vague intents, and near-searches against non-searches, under the policy below | «שולחן?», «משהו לסלון באזור», «איך מוכרים רכב» (not a search) |

### Per-field quotas (top of the list; full list in the JSON)
Target: every field ≥ **400** examples across train + `extra_train` + this batch; the high-error-rate fields ≥
**700**. "+new" is the minimum this batch must add.

| field | have | target | +new | why |
|---|---|---|---|---|
| נדל״ן/רחוב | 157 | 700 | **543** | high error rate |
| יד_שנייה/מותג | 216 | 700 | **484** | high error rate |
| נדל״ן/ריהוט | 241 | 700 | **459** | high error rate |
| יד_שנייה/אחסון_GB | 251 | 700 | **449** | high error rate |
| רכב/טווח_חשמלי_ק״מ | 258 | 700 | **442** | high error rate |
| יד_שנייה/שנת_ייצור | 260 | 700 | **440** | high error rate |
| רכב/תתי_דגמים | 284 | 700 | **416** | high error rate |
| נדל״ן/שכונה | 404 | 700 | **296** | high error rate |
| נדל״ן/מרפסות | 421 | 700 | **279** | high error rate |
| רכב/טעינה_מהירה | 149 | 400 | 251 | rare |
| רכב/טסט_עד | 167 | 400 | 233 | rare |
| נדל״ן/גישה_לנכים | 174 | 400 | 226 | rare |
| … 22 more fields | | | | see the JSON |

## Labelling rules (existing, plus three clarifications)
The existing rules apply unchanged (README of `y2_dataset_*`). In short:
- **Numbers and dates** are written in the span, never calculated.
- **Free-text values** are canonical (city spellings, no «שכונת» prefix), and a city is never inferred from a street.
- **Brands:** on a sub-category's list → `מותגים`; otherwise `מותג`.
- **A powertrain named in the query** goes in `סוג_דלק`.
- **Selling, renting out, advice, site support and chit-chat** are not searches (category null).

Clarifications, from inconsistencies found in the current data:
1. **Canonical names:** `עיר` / `שכונה` / `רחוב` take the **gazetteer's canonical spelling verbatim**, including its
   hyphenation. Never include units or words around the name: in «בשכונת הדר 120 מ"ר» the value is «הדר», not «הדר 120 מ"ר».
2. **Property type:** one value, the **most specific** one. «דירת גג» → `סוגי_נכס: גג/טיפוס`, not
   [דירה, גג/טיפוס]. A list only when the user genuinely accepts alternatives («דירה או קוטג׳»).
3. **Storage and memory:** sizes written in digits keep their number («500 גיגה» → 500). Sizes written in words are
   powers of two («חמש מאות ג'יגה» → 512, «טרה» → 1024). TB × 1024.

**Open decision (owner: product):** should a hybrid powertrain also set `סוגי_רכב: רכב היברידי/חשמלי`?
- **Default:** keep the current rule (`סוג_דלק` only).
- **If it changes:** change it in the generator rule, relabel all existing exports with a deterministic script, and
  tell the parser project. It then switches on `TAGGER_HYBRID_VEHICLE_TYPE`.

**Non-search policy (B7):** a query is a search when it names something to find, or a constraint on it, even
vaguely («משהו לסלון באזור» → `יד_שנייה`). It is **not** a search when it asks how to do something, sells or rents
out, or is chit-chat («איך מוכרים רכב», «מה המצב»).

## Leakage and dedupe
**Dedupe against everything:**
- `training/training_bert/data/{train,val,test}.jsonl`
- `training/training_frozen_fusion/data/**` (split1, `extra_train`, the routing batch)
- the calibration batch `y2_routing_v2`, once it exists

**Rules:** same as before.
- No identical normalized query, no near-duplicate (character 4-gram Jaccard ≥ 0.6), no contained query.
- Rephrasings, near-duplicates and identical non-empty labels share a `unit`.

**Never put validation or test queries in prompts or as few-shot examples.**

## What the parser project will check on delivery
1. **Schema:** every row validates; spans' `text` equals `query[start:end]`; labels are in the v2 schema and the
   taxonomy.
2. **Quotas:**
   - each bucket within ±10% of its target
   - every field's `min_new_rows` met
   - typo share ≥ 40%
3. **Leakage:** zero matches against all the files above.
4. **Spot check:** about 100 random rows reviewed blind. More than 3% wrong sends the batch back.

Then the batch is merged with train + `extra_train`, the 3-seed ensemble is retrained (~1 h on a 3090), and the
result is compared on val_eval / val_calib, then checked on test once.

## Related
- The **calibration batch** `y2_routing_v2` (calibration only, ~6k rows, ≥ 600 random gold, ≥ 300 written by people)
  is specified in [../data_generation_plan.md](../data_generation_plan.md), section A. It's independent of this one,
  and both can run in parallel.
