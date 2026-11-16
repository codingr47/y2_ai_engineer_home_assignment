# Taxonomy issue: point vs range field names (`מחיר` vs `טווח_מחיר`, …)

**Status:** open; needs a decision from the taxonomy owner.
**Taxonomy:** `yad2_search_taxonomy.json` v1.0 (2025-09-30), sha256 `d4786a63…` (as in
`eval/dataset/inventory/field_inventory.json`).
**Evidence:** `python docs/point_range_evidence.py` (train/val/test of `training/data/split1`, 10,394 rows).

## Summary
- For 7 numeric concepts the taxonomy defines two fields: a point field under `מאפיינים_כלליים` and a range filter
  under `פילטרים_נוספים`.
- It never says which one a bounded value ("עד 2 מיליון", "בין 3 ל-4 חדרים") belongs in.
- The point fields are not restricted to exact values, and the data stores bounds in them using the same
  `{min, max}` shape as the filters. So the two names are interchangeable for every bounded value.
- The labelled data uses both. A bounded value goes to the point field in 57% of cases and to the range field in
  43%. The choice comes from how the generator sampled fields, not from the wording.
- A model cannot learn a choice that isn't encoded in the text, so field-name accuracy for these concepts is
  capped by label noise.

---

## 1. What the taxonomy defines

### 1.1 The seven point/range pairs
| domain | point field (`מאפיינים_כלליים`) | point spec | range field (`פילטרים_נוספים`) | range spec |
|---|---|---|---|---|
| נדל״ן | `מחיר` | `{"טיפוס":"מספר","מטבע":"ש״ח"}` (no bounds) | `טווח_מחיר` | `{"min":0,"max":50000000}` |
| נדל״ן | `מס׳_חדרים` | `{"טיפוס":"מספר","טווח":[1,12]}` | `טווח_חדרים` | `{"min":1,"max":12}` |
| נדל״ן | `מ״ר_בנוי` | `{"טיפוס":"מספר","טווח":[10,1000]}` | `טווח_מ״ר` | `{"min":10,"max":1000}` |
| נדל״ן | `קומה` | `{"טיפוס":"מספר","טווח":[0,60]}` | `טווח_קומה` | `{"min":0,"max":60}` |
| רכב | `מחיר` | `{"טיפוס":"מספר","מטבע":"ש״ח"}` (no bounds) | `טווח_מחיר` | `{"min":5000,"max":2000000}` |
| רכב | `שנה` | `{"טיפוס":"מספר","טווח":[1980,2025]}` | `טווח_שנה` | `{"min":1980,"max":2025}` |
| רכב | `ק״מ` | `{"טיפוס":"מספר","טווח":[0,1000000]}` | `טווח_ק״מ` | `{"min":0,"max":1000000}` |

### 1.2 Numeric fields with no range twin
| domain | point fields without a `טווח_*` twin |
|---|---|
| נדל״ן | `סה״כ_קומות`, `מ״ר_מגרש`, `מרפסות`, `חניה`, `ארנונה_חודשית` |
| רכב | `יד`, `נפח_מנוע_סמ״ק`, `הספק_כ״ס`, `טווח_חשמלי_ק״מ`, `מספר_בעלים_קודמים` |
| יד_שנייה | `מחיר`, `שנת_ייצור`, plus sub-category numbers (`זיכרון_RAM`, `אחסון_GB`, `גודל_אינצ׳`, per-sub-category `מחיר`) |

Only 7 of about 22 numeric fields have a range twin.

### 1.3 The only range-related rule in the taxonomy
`כללי.כללי_ניקוי_קלט` contains `"זיהוי טווחים (עד/בין/מעל/מתחת)"` ("detect ranges"). It is an input-cleaning step
and does not say which output field a detected range goes to.

---

## 2. Inconsistencies in the taxonomy itself

**T1. Two fields per concept, no selection rule.**
- The point field's spec doesn't restrict it to exact values. The range field's spec doesn't say it is
  *the* place for bounds.
- Nothing states "bounded → `טווח_*`" or "point fields accept `{min, max}`". Both readings are valid.

**T2. Partial, asymmetric coverage.**
- Twins exist for 4 real-estate concepts, 3 vehicle concepts and no second-hand concepts.
- `סה״כ_קומות`, `חניה`, `מרפסות`, `יד`, יד_שנייה `מחיר` and the others must carry bounds in the point field,
  because no other field exists for them. So "point fields hold bounds" is already the de facto rule for 15 of
  the ~22 numeric fields.

**T3. Pair names don't line up.**
- `טווח_חדרים` pairs with `מס׳_חדרים`, not `חדרים`.
- `טווח_מ״ר` doesn't say *which* area it is: `מ״ר_בנוי` (built) or `מ״ר_מגרש` (lot). Only the matching bounds
  (10–1000) suggest built area. Lot-area bounds have no filter.

**T4. "טווח" means three different things.**
- (a) Inside a point spec, `"טווח": [lo, hi]` means *valid value bounds* (e.g. `קומה`: `[0, 60]`).
- (b) As a field-name prefix, `טווח_X` means *a range filter on X*.
- (c) In `טווח_חשמלי_ק״מ`, it means an *EV's driving range*. That is a point field whose name starts with
  `טווח_`, so it looks like a filter but isn't one.

**T5. Different schemas and bounds for the same concept.**
- Point fields use `{"טיפוס": …, "טווח": […]}`; filters use bare `{"min", "max"}` with no type or currency.
- `מחיר` has no bounds in either domain, while `טווח_מחיר` has them. The vehicle filter has a 5,000 minimum
  that the point field doesn't enforce.

**T6. Three year concepts for vehicles.**
- `שנה` (point) and `טווח_שנה` (filter).
- `יצרנים.<manufacturer>.שנות_ייצור`: a per-manufacturer catalogue span such as `[1995, 2025]`. The inventory
  lists it as a conditional `range` field, and the data uses it as a query field.
- יד_שנייה adds a fourth: `שנת_ייצור`, with no filter.

---

## 3. How the labelled data uses the pairs (measured)

### 3.1 Bounded values split between the names
| pair | bounded → point | bounded → range | exact → point | exact → range |
|---|---|---|---|---|
| נדל״ן `מחיר` / `טווח_מחיר` | 116 | 80 | 57 | 0 |
| נדל״ן `מס׳_חדרים` / `טווח_חדרים` | 103 | 76 | 55 | 0 |
| נדל״ן `מ״ר_בנוי` / `טווח_מ״ר` | 80 | 79 | 55 | 0 |
| נדל״ן `קומה` / `טווח_קומה` | 86 | 79 | 58 | 0 |
| רכב `מחיר` / `טווח_מחיר` | 86 | 61 | 35 | 0 |
| רכב `ק״מ` / `טווח_ק״מ` | 69 | 58 | 42 | 1 |
| רכב `שנה` / `טווח_שנה` | 42 | 12 | 44 | 0 |
| **total** | **582 (57%)** | **445 (43%)** | 346 | 1 |

| quality tier | bounded → point | bounded → range |
|---|---|---|
| gold: hand-written or checked (test only) | **89 (77%)** | 27 (23%) |
| silver: generated, auto-approved | 493 (54%) | 418 (46%) |

- Exact values always use the point field, apart from one `טווח_ק״מ` row with a plain integer.
- **The ambiguity is entirely in bounded values**, and the two tiers follow different habits: human annotators
  mostly use the point name, generated labels are close to a coin flip.

### 3.2 Same wording, different label (spans quoted verbatim, typos included)
| wording (silver) | labelled as | same pattern labelled as |
|---|---|---|
| "עד 275,000 ש"ח" (syn_10176) | רכב `מחיר` | "עד 375 אלף שקל" (syn_10202) → `טווח_מחיר` |
| "עד חמישה מיל,יון וחצי" (syn_10174) | נדל״ן `מחיר` | "עד 4450 אלף שקל" (syn_10179) → `טווח_מחיר` |
| "לא מעל קומה 13" (syn_10138) | `קומה` | "עד קמה 20 לכל היותר" (syn_10022) → `טווח_קומה` |
| "עד 55K קמ" (syn_10163) | `ק״מ` | "עד 125000 ק״מ" (syn_10292) → `טווח_ק״מ` |

### 3.3 Same value shape
- When a point field carries a bound, its value is `{"min": …}`, `{"max": …}` or `{"min": …, "max": …}`, the
  same object the `טווח_*` fields hold.
- So nothing in the value says which field was meant; only the key differs.

### 3.4 Fields without twins carry bounds in the point field
Bounded share of values in fields with no `טווח_*` twin:

| field | bounded share |
|---|---|
| יד_שנייה `מחיר` | 84% |
| `ארנונה_חודשית`, `חניה` | 75% |
| `מרפסות` | 75% |
| `סה״כ_קומות` | 74% |
| `מ״ר_מגרש` | 73% |
| `יד` | 72% |
| `נפח_מנוע_סמ״ק`, `שנת_ייצור` | 71% |
| `זיכרון_RAM`, `אחסון_GB`, `גודל_אינצ׳`, `הספק_כ״ס`, `מספר_בעלים_קודמים`, `טווח_חשמלי_ק״מ` | 76–77% |

The dataset already relies on "point fields hold bounds" everywhere a twin doesn't exist.

### 3.5 Vehicle years: a three-way split
| field | exact | bounded, manufacturer named | bounded, no manufacturer |
|---|---|---|---|
| `שנה` | 44 | 16 | 26 |
| `שנות_ייצור` | 0 | **79** | 0 |
| `טווח_שנה` | 0 | 3 | 9 |

A bounded vehicle year can land in any of three fields. `שנות_ייצור`, the manufacturer's catalogue span in the
taxonomy, is used as a *query* constraint whenever a manufacturer is named.

---

## 4. Root cause in the data generator
These references are from `../y2_ai_synthetic_generator/synthetic_generation_app/` (read-only).

1. **The generator treats the pairs as aliases.** `semantics.py` defines `ALIAS_GROUPS` and says "a query can
   legitimately map to either":
   `("מחיר","טווח_מחיר")`, `("שנה","טווח_שנה","שנות_ייצור")`, `("ק״מ","טווח_ק״מ")`,
   `("מס׳_חדרים","טווח_חדרים")`, `("מ״ר_בנוי","טווח_מ״ר")`, `("קומה","טווח_קומה")`, plus `("מותגים","מותג")`.
2. **The planner samples either name as a separate field.** Inventory entries are sampled independently by weight
   (`planner.FIELD_WEIGHTS`, e.g. `"מחיר": 5`, `"טווח_מחיר": 1`); `RANGE_BASE` only borrows the base field's value
   profile. The label key is whichever entry was drawn, and the query text is then written from the values, so the
   wording carries no trace of the choice.
3. **The extractor is told the opposite.** The blind-extraction prompt (`prompts.py`) says "Prefer base fields
   (מחיר, שנה, ק״מ, מס׳_חדרים, מ״ר_בנוי, קומה) over the טווח_ variants."
4. **Silver approval can't see the disagreement.** Labels are compared after `canon_key()` merges the aliases. The
   generator's own test asserts `{"מחיר": {"max": 70000}} == {"טווח_מחיר": {"max": 70000}}`. A planner label of
   `טווח_מחיר` and an extractor answer of `מחיר` therefore count as agreement, and the row is approved with the
   planner's key.

**Result:** the silver labels' choice between point and range name is essentially random. Gold annotators mostly
follow the extractor's "prefer base fields" convention (77%). Neither follows a rule the taxonomy states.

---

## 5. Impact on the models and the evaluation (measured in `training/`)
- **Experiment 1 (77 fields):** twin pairs account for **19% of all field errors** on val_eval, and the 5 most
  frequent confusions are all twin swaps (e.g. `טווח_ק״מ→ק״מ`, `טווח_מחיר→מחיר`).
  See `training/experiment_1/experiment_analysis/REPORT.md`.
- **Merging each pair into one slot for presence detection** (70 slots) gives **+1.1 points** of slot exact-set
  on test (family-bootstrap 95% CI +0.4 to +2.0).
- **Field-name accuracy is capped by the labels, not the model.** Given the gold operator, the taxonomy-literal
  rule (bounded → `טווח_*`) reproduces the gold name for 63% of val_eval twin spans, 46% of test spans and
  **37% of gold test spans**. Always choosing the point name reproduces 61%, 76% and **80%**.
  See `training/comparison/REPORT.md` §6.
- **Not yet covered:** the slot merge does not include `שנות_ייצור`, so the vehicle-year ambiguity (§3.5) is
  still inside the label space.

---

## 6. Options (decision needed)
| option | rule | data change | consistent with | cost |
|---|---|---|---|---|
| **A. Point fields carry bounds** | Every numeric field holds `number` or `{min, max}`; `טווח_*` is retired, or kept only as an alias the output layer can emit. | Relabel the 445 bounded `טווח_*` spans (a mechanical key rename), and fold `שנות_ייצור`-as-query into `שנה`. | §3.4 (15 of 22 fields already work this way), gold annotators (77%), the extractor prompt, the value shape (§3.3) | lowest; no new taxonomy fields |
| **B. Taxonomy-literal** | bounded → `טווח_*`, exact → point field | Relabel the 582 bounded point spans for the 7 concepts. Decide what to do for the 15 fields without twins (add `טווח_*` for all, or accept an asymmetric rule). | the taxonomy's filter section | high; either many new filters or a rule that differs by field |
| **C. Both are valid output names** | Pick one canonical name for scoring; the API accepts or emits either. | None. Evaluation canonicalises; the output layer maps via operator if a downstream search API needs `טווח_*`. | the generator's alias semantics | low, but the taxonomy stays ambiguous for anyone else using it |

**Recommendation: A.** The data, the human annotators and the extraction guidance already behave this way. It
removes T1, T2 and T4(b) in one step, and value `{min, max}` already carries the operator. If a downstream Yad2
search endpoint needs filter keys, derive `טווח_*` deterministically from the operator in the taxonomy layer
(`training/shared/taxonomy_rules.resolve`).

Independent of the option chosen:
- clarify `טווח_מ״ר` (built vs lot) (T3)
- rename the `"טווח"` value-bounds key inside point specs, e.g. to `"ערכים_חוקיים"` (T4a)
- give price fields explicit bounds consistent with their filters (T5)
- state whether `יצרנים.<manufacturer>.שנות_ייצור` is catalogue metadata or a query field (T6)

## 7. What the code does today
- `training/shared/slots.py` merges the 7 pairs into one slot each for detection. Generated labels can't separate
  the two names, so this is the honest target.
- `training/shared/taxonomy_rules.Resolver.resolve(slot, local_span, operator)` picks the output name.
  - With **no operator** (deployable today) it returns the train-majority name, which is the point field for all 7
    slots. That matches option A.
  - With an operator it applies the taxonomy-literal rule (option B), so either decision is one switch.
  - It only ever uses the field's own span, never the whole query.
- Under option A, the resolver's default stays and the operator branch would map to `טווח_*` only for an external
  API. Under option B, the train/val/test labels must be relabelled before strict metrics mean anything.
