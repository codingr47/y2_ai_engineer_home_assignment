# Data generation plan: calibration batch + training batch (for the generator project)

**Context.** The Fine-Tuned Tagger (`training/training_bert/experiment1–3`) reaches 85.1% JSON exact on test. Its gate,
with Frozen Fusion as a second opinion, answers 84.5% of queries locally at 92.9% accuracy, at the 0.95 target. Two
data gaps remain; neither is the labels, which a blind audit put at about 1–3% error:
1. **Calibration is too small to certify 95%.** On the routing batch, a properly certified 95% exists only at 39%
   coverage. A val-calibrated gate *measures* 95.5% at 84% coverage on the routing audit, but 611 rows can't prove it.
2. **Training data is thin where the model is weak.** On test: typo queries 78.0%, real estate 80.5%, gold-style
   phrasing 79.5%. The learning curve still gains about 1 point per 5k rows.

Generation cost, from the earlier batches: about **$4.2 per 1k rows** (routing batch $14.88, extra batch $17.30).

**Both batches:**
- Same export schema as `y2_dataset_*` / `y2_routing_*`: `id, query, domain, expected, metadata, family, unit,
  quality, spans`.
- Labels in the v2 schema; `spans` with explicit character offsets for every field the query states.
- **Leakage rules as in the routing batch:**
  - no row may match any existing row — same normalized query, near-duplicate (4-gram Jaccard ≥ 0.6), or contained
    query
  - matched against all of `training/training_bert/data/{train,val,test}.jsonl`, `training/training_frozen_fusion/data/**` and the
    other batch
  - rows linked by rephrasing, near-duplication or identical non-empty labels share a `unit`
- A `manifest.json` with seed, generator and config version, ids, gold ids, and the SHA-256 of every input
  deduplicated against.
- **Decide the hybrid convention first,** and apply it to both batches:
  - Today a named powertrain goes only in `סוג_דלק`.
  - If Yad2's search expects `סוגי_רכב: רכב היברידי/חשמלי` too, change the generator rule.
  - Then also relabel the existing splits with a deterministic script, and switch on
    `SpanNormalizer.hybrid_vehicle_type` / `TAGGER_HYBRID_VEHICLE_TYPE`.

---

## A. Calibration batch `y2_routing_v2` — calibration only

**Purpose:** fit / threshold / audit the gate. **Never train, fine-tune, early-stop or tune anything else on it**
(same README rule as the routing batch).

| item | target | why |
|---|---|---|
| rows | **~6,000** (~5,000 units) | about 600–800 accepted units in each of the threshold and audit parts at ~80% coverage, enough for a 95% Wilson lower bound when accepted accuracy is ≥ 96.3% |
| partition | fit 50% / threshold 25% / audit 25%, by `unit`, seed fixed in the manifest | larger threshold and audit parts than 60/20/20; the fit part is plentiful |
| protocol change (parser side) | threshold part lower bound ≥ **target** (0.95), not target + 0.01; the audit must also reach ≥ 0.95 | the +0.01 margin alone needs ~10k accepted units |
| gold (person-checked) | **≥ 600**, a *random* sample, not only disagreements | today's gold is mostly LLM-disagreement cases, hard by selection |
| human-written queries | **≥ 300** | today 0; the gate must be measured on real phrasing |
| Claude-written share | ≤ 10% and tagged | if the LLM fallback is a Claude model, its own phrasing flatters it |
| domain mix | production-like: real estate ~40%, vehicles ~30%, second-hand ~25%, non-search ~5–10% | the gate's coverage is a traffic-weighted number |
| language register | typo ~35–40%, slang ~30%, standard ~30% | matches the earlier batches and test |
| ambiguity | keep genuinely ambiguous queries (e.g. «משהו מעץ», «מתחת לחמישים אלף») and label them as people would | these are the queries the gate must learn to route away |

**Cost:** about 6k × $4.2/1k ≈ **$25**, plus review time for the 600 gold rows.

## B. Training batch `y2_train_v3` — training only

**Purpose:** more data where the tagger is weak. It's used for training only: never for selection, calibration or
test.

| item | target | why |
|---|---|---|
| rows | **~8,000** (+ the unused `extra_train`, 3,188 rows) | about +11k rows over today's 17k; the learning curve suggests +1.5–2 points |
| domain mix | real estate **40%**, vehicles 30%, second-hand 25%, non-search 5% | real estate is the weakest test domain (80.5%) |
| language register | typo **45%**, slang 30%, standard 25% | typo is the weakest slice (78.0%) |
| typo kinds | glued words («בראשוןלציון»), final-letter misuse, Latin/digit lookalikes («בחdרה», «טלפ1ן»), in-word punctuation, doubled letters, keyboard-adjacent swaps | the kinds the runtime cleaner can't fix are the ones the model must learn |
| phrasing | indirect and story-like requests, implied fields (item word → sector / sub-category), self-corrections («5 מיליון, סליחה 4.5»), mixed operators | what gold rows look like |
| quality | silver (bulk-approved after a blind extraction agreed) is fine; a ~2% random human spot check | the audit measured ~1–3% error for silver |
| spans | required for every explicit field, with exact offsets; derived values without a span are fine (as today) | the tagger learns from spans |

**Cost:** about 8k × $4.2/1k ≈ **$34**.

## Hand-off to the parser project
- **Delivery:** both exports go under `training/training_frozen_fusion/data/` (or `training/training_bert/data/batches/`), each with its
  README stating its role. The existing files are never modified.
- **Training batch:** retrain the 3-seed ensemble on train + `extra_train` + `y2_train_v3` (~1 h on a 3090,
  ~$0.25), re-score val_eval / val_calib, then check test once.
- **Calibration batch:** rebuild the gate with `scripts/build_tagger_gate.py --calibration <routing_v2>` on the 50 /
  25 / 25 partition, then check test once.
