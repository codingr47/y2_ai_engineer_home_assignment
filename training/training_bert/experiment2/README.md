# Fine-Tuned Tagger — Experiment 2: Frozen Fusion as a second opinion + runtime typo cleanup (2026-11-17)

**Base system.** Experiment 1's final configuration, unchanged:
- 3-seed DictaBERT-base ensemble (E6)
- normalizer with the number-word fixes
- extractor fallback `presence@0.5`

No model was retrained, and no data file was changed. Everything runs at inference, locally on CPU.

**Result (test, scored once):**
- **Runtime cleanup:** JSON exact **84.8% → 85.1%**.
- **The gate with Frozen Fusion's opinion,** at the pre-registered 0.95 target: **84.5%** of queries answered locally at **92.9%** accuracy. Experiment 1's gate: 77.2% at 92.4%.
- At the user's volumes (~33M searches a month, ~$0.10 per 1k LLM calls), LLM cost goes from about **$740 to about $505** a month (LLM only: about $3,250).

## 2A — Frozen Fusion as a second opinion for the gate
**Frozen Fusion's predictions:** `ff_predictions.py` → `results/ff_<split>.jsonl`.
- It runs on the raw query.
- It uses the shipped `src/artifacts` parser, with the tolerant code binding from `frozen_fusion_baseline.py`.

**New gate signals** (`gate_ff.py`): same JSON as the tagger, same domain, same field set, Frozen Fusion's own gate
score, whether it parsed at all.

**Protocol:** the same as experiment-1 E12.
- Fit: logistic regression on val_eval.
- Threshold: val_calib half A (by family), on a one-sided 95% Wilson bound ≥ target + 0.01.
- Audit: half B.
- Test: once.

**Agreement:** the two systems agree on 62–67% of val rows and 77% of test rows. When they agree the tagger is right
96%, when they disagree 80–81%.

| gate (cleaned queries) | target | threshold | audit: answered locally / accuracy (lower bound) | test: answered locally / accuracy |
|---|---|---|---|---|
| tagger only | 0.95 | 0.918 | 80.6% / 95.5% (0.944) | 73.8% / 93.4% |
| **tagger + Frozen Fusion** | **0.95** | **0.818** | **86.1% / 95.5% (0.943)** | **84.5% / 92.9%** |
| tagger + Frozen Fusion | 0.96 | 0.884 | 80.5% / 96.3% (0.952) | 79.6% / 94.2% |
| tagger + Frozen Fusion | 0.97 | 0.906 | 76.0% / 96.8% (0.957) | 76.9% / 94.5% |

- **On raw queries the picture is the same.** Tagger only at 0.95: 77.2% / 92.4% on test. Tagger + Frozen Fusion:
  85.9% / 92.0%. Full curves: `results/gate_ff_report*_with_test.json`.
- **A simple "accept iff they agree" rule is unstable.** Its coverage is 85% on one val_calib half and 53% on the
  other, because Frozen Fusion is weak on v2 rows and the halves differ in v2 share. The learned gate handles this.
- **Caveat:** the 0.96 / 0.97 rows pass a 95% lower bound on the audit half, but they were noticed after seeing the
  audit. The pre-registered target is 0.95.

## 2B — runtime query cleanup (`training/training_bert/common/clean.py`)
**What it is:** `QueryCleaner.clean(query)` runs before the tagger at inference. The data files are never touched.

**When a word is rewritten:** only when a candidate is clearly the more frequent word. That means
freq ≥ 5 × freq(original) + 5, with frequencies from train queries plus lexicon surface words
(`artifacts/clean_vocab.json`, regenerable). Every rule except final-letter fixes also requires the original to be
rare as written (≤ 3 occurrences).

**Rules:**

| rule | what it fixes | examples |
|---|---|---|
| invisible marks | removed | |
| final letters | the wrong final form | «אלפ», «דברימ», «מחףש» |
| repeated letters | doubled final letters (never Hebrew) and runs of 3+ | «בחולוןן», «דחוףףף» |
| Latin / digit lookalikes | one stray character inside a Hebrew word | «בחdרה», «טלפ1ן» |
| punctuation inside a word | `. , _ *` only | «סמס.ונג». Never ' " ׳ ״, which mark abbreviations. |
| glued words | two known words joined, both halves ≥ 3 letters | «פלאפוןסלולרי», «בקרייתאתה» |

**Protected:** «לפטופ», «כלל», «מוכנים», «סמ"ק», «ב.מ.וו», «יד_שנייה».

**Measurement** (`clean_eval.py`): only changed rows are re-predicted; for the others the input is identical.

| split | rows rewritten | JSON raw → cleaned | changed rows: fixed / broken |
|---|---|---|---|
| val_eval | 123 / 1,617 | 91.2% → 91.7% | 9 / 1 |
| val_calib | 128 / 1,819 | 90.0% → 90.5% | 11 / 3 |
| test | 132 / 2,181 | 84.8% → **85.1%** | 10 / 2 |

The typical breakage is a bad split («משוממש» → «משו ממש»). The rules were not tuned further on val.

## Cost reading
At the 0.95 target, using the user's figures (40% of traffic = 13.2M searches = $1,300 a month):

| configuration | LLM calls a month | LLM cost a month |
|---|---|---|
| LLM only | 33M | ~$3,250 |
| Experiment 1 gate | 7.5M | ~$740 |
| **Experiment 2** | **5.1M** | **~$505** |

If Luna (reported 89.36% on test, not measured here) keeps its rate on the queries the gate sends it, the routed
system is about 0.845 × 92.9% + 0.155 × 89.4% ≈ **92.4%** on test.

## Reproduce
```bash
python -m training.training_bert.experiment2.ff_predictions               # Frozen Fusion on val_eval / val_calib / test (CPU)
python -m training.training_bert.experiment2.clean_eval [--with-test]     # 2B (CPU; re-predicts only rewritten rows)
python -m training.training_bert.experiment2.gate_ff [--clean] [--with-test]   # 2A gates
```
Inputs: experiment-1 checkpoints `experiment1/runs/e6_lr5e-5_s*/model.pt` and the E12 predictions.
