# Fine-Tuned Tagger — Experiment 3: arbiter, routing-batch calibration, label audit (2026-11-17)

**Base system:** Experiment 2's final: cleaned queries → 3-seed tagger ensemble → normalizer + extractor fallback,
with Frozen Fusion as a second opinion. No retraining, no data file changed. **Test was not used** in this
experiment.

**Results:**
1. **Arbiter:** a per-field arbiter between the tagger and Frozen Fusion doesn't generalize. Negative.
2. **Routing-batch calibration:**
   - A gate calibrated on the routing batch, with its own protocol, **certifies 95% only at 39% coverage**. The batch
     is too small to prove more.
   - Experiment 2's val gate measures **95.5% on the routing audit at 84% coverage**, but can't certify it (lower
     bound 0.936).
3. **Label audit:**
   - A blind human audit of 120 random val labels: **0.7% wrong**, about 2–3% counting the reviewer's
     "missing field" notes. The data is not the bottleneck.
   - The hybrid → vehicle-type rule was added as an off-by-default product rule.

The planned gold-gap analysis was **dropped**. The routing batch's README reserves it for calibration only, and no
other gold data exists outside test.

## 1. Per-field arbiter (`arbiter.py`) — negative
**Ceiling:** if, for every field where the two systems disagree, we always picked the right one, val_eval would gain
**+3.0** points JSON and val_calib +2.8. In about two-thirds of those rows Frozen Fusion's whole parse is right.

**Model:** one logistic model scores every disputed field. A disputed field is one of:
- **add:** Frozen Fusion has a field the tagger lacks
- **drop:** the tagger has a field Frozen Fusion lacks
- **switch:** both have it, with different values

Signals: kind, field type, domain, the tagger's presence probability and confidence, Frozen Fusion's gate score, the
field counts.
- Fit on val_eval: 620 decisions; Frozen Fusion is right in 9.8% of them.
- Threshold: on 5-fold out-of-fold scores (folds by family). The curve peaked at 0.95.

| split | tagger | + arbiter | rows fixed / broken |
|---|---|---|---|
| val_eval (in-sample) | 91.7% | 92.1% | 13 / 6 |
| **val_calib** | 90.5% | **90.4%** | 5 / 7 |

**Verdict:** the arbiter can't tell which system is right on a disputed field. It wasn't tuned further on val_calib.
Frozen Fusion stays a second opinion for the gate, where it pays (Experiment 2).

## 2. Gate calibration on the routing batch (`routing_predict.py`, `calibrate_routing.py`)
**The batch's role, unchanged:** calibration only, per its README.
- **Read as is:** `routing.jsonl`'s SHA-256 matches its manifest (`8f5fd112…`).
- **Partition:** the ids already recorded in `docs/confidence_audit.json` (fit 1,846 / selection 602 / audit 611
  rows, by unit, seed 42). Nothing was redefined.
- **Protocol:** Frozen Fusion's own functions in `scripts/calibrate_confidence.py`:
  - scaler + logistic regression on fit
  - threshold = the largest coverage on selection whose Wilson lower bound, **counted over accepted units**, reaches
    target + 0.01
  - one audit

**Tagger accuracy on the batch:** **87.3%** JSON exact (silver 87.9%, **gold 79.7%**). That's between val (90.5%)
and test (85.1%), and the gold gap from test shows up here too.

| gate | threshold | audit: answered locally / accuracy (lower bound over units) | 95% certified |
|---|---|---|---|
| tagger only | none reaches 0.96 on selection | – | ✗ |
| **tagger + Frozen Fusion** | **0.972** | **39.1% / 97.9% (0.957)** — gold rows 23.5% / 91.7%, silver 40.5% / 98.2% | ✓ |
| Experiment 2 val gate, applied unchanged | 0.818 | 84.3% / 95.5% (0.936), 515 accepted / 431 units | ✗ |

**Reading:**
- **Certifying 95% needs more units, not a better model.** With about 490 units in the selection part, proving
  ≥ 0.96 requires about 98% measured accuracy, so the threshold climbs to 0.972.
- **Experiment 2's val-calibrated threshold already measures 95.5%** at 84% coverage on this test-like audit. The
  shift from val to routing is smaller than from val to test (92.9%).
- **The batch is short of its own target:** 236 of 300 gold rows, 0 human-written queries. About 2–3× the units would
  be needed to certify 95% at about 80% coverage.

## 3. Label audit (`audit/`)
**Sample** (`audit/sample.jsonl`): 120 rows, simple random from val only (test untouched).
- **80** from the 2,041 bulk-approved silver rows. Each was confirmed `bulk: true` with no human verdict in the
  generator's `synthetic_reviews.jsonl`.
- **40** from the 1,395 synthetic v2 rows.

**Method:** reviewed blind (label only, no model output) in `audit/audit.html` by the user. Verdicts are in
`audit/audit_verdicts.json`; `audit/score_audit.py` → `audit/audit_result.json`.

| pool | n | wrong | unsure | error rate (95% CI) |
|---|---|---|---|---|
| silver, bulk-approved | 80 | 1 | 2 | 1.2% (0.2–6.7%) |
| synthetic v2 | 40 | 0 | 5 | 0.0% (0.0–8.8%) |
| **val-weighted** | 120 | | | **0.7%** |

**The unsure notes:**
- **2 are missing fields,** effectively wrong: «...בקרית מלאכי» without the city; a prompt-injection row without
  «כורסה».
- **2 are the hybrid convention:** «היברידי» without `סוגי_רכב`.
- **3 are genuinely ambiguous queries:** «משהו מעץ»; «מתחת לחמישים אלף», price or mileage?; real estate inferred
  from price alone.
- **The one wrong row:** «פיקסל» missing as the model.

**The review log in context:** 92–93% of the generator's rows were bulk-approved because the generator and a blind
LLM extraction (Haiku) agreed exactly. People reviewed mostly the *disagreements*, where about 14% needed a fix or a
drop.
- That 14% is the error rate of hard, pre-selected cases, not of silver.
- It also explains much of the gold gap: gold rows are hard by construction.

**Verdict:** label error is about 1–3%, so the ceiling on these labels is about 97%. The remaining gap (val 90.5%,
test 85.1%, Luna 89.4%) belongs to the model and to query ambiguity.

### Hybrid → vehicle-type rule (product rule, off by default)
**The rule:** `normalize.hybrid_vehicle_type`: `סוג_דלק` ∈ {היברידי, היברידי נטען} adds
`סוגי_רכב: רכב היברידי/חשמלי`, unless a vehicle type is already set.
- Off by default (`SpanNormalizer.hybrid_vehicle_type`).
- The dataset rule is the opposite (a powertrain goes only in `סוג_דלק`), and 92% of train hybrid rows follow it.

**Turned on against the current labels:**
- val_eval: 49 rows change, JSON 91.7% → 89.2%
- val_calib: 16 rows, 90.5% → 89.9%

**Next:** turn it on together with the same change to the generator's labelling rule.

## Process note
- **What happened:** a command the user rejected (a routing loader in `common/data.py` plus a "dev slice" split
  file) had already run. Nothing used it.
- **Revert:** `data.py` was restored to the committed version, and `artifacts/routing_split.json` was deleted. The
  routing file's hash was confirmed against its manifest.

## Reproduce
```bash
python -m training.training_bert.experiment3.arbiter                # step 1 (val only)
python -m training.training_bert.experiment3.routing_predict        # tagger + Frozen Fusion on the routing batch (CPU, ~15 min)
python -m training.training_bert.experiment3.calibrate_routing      # step 3 (no test)
python -m training.training_bert.experiment3.audit.build_page       # audit page; then score_audit after the review
```
