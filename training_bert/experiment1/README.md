# Fine-Tuned Tagger — Experiment 1: DictaBERT multi-task tagger (2026-11-17)

The first experiment of the **Fine-Tuned Tagger** phase. It follows **Frozen Fusion**, the name for all work up to
2026-11-16: frozen E5 + TF-IDF, LR heads, the rule/lexicon extractor and the confidence gate.

**Result.** A fine-tuned DictaBERT-base is ensembled over 3 seeds and decoded through the existing normalizer, with
Frozen Fusion's extractor as a fallback.
- **test: 84.8% JSON exact** (Frozen Fusion: 75.4%, same 2,181 rows, test scored once)
- **val_eval: 91.2%** (Frozen Fusion on the same rows: 67.8%)
- **val_calib: 90.0%**

A confidence gate set for 95% answers 77% of test queries locally, at **92.4%** accuracy. So 95% is *not* certified:
the audit lower bound is 0.939.

Every number below is read from `runs/` by `summarize.py` into `results/summary.json`.

---

## 1. Context and goal
- **Where Frozen Fusion stalled.** It plateaued at 83.8% val_eval and 75.4% test. Its gate accepted 50% of test
  queries at 86.5% accuracy. More data of the same kind gave diminishing returns (Exp7).
- **The goal here.** Fine-tune a Hebrew encoder end to end for:
  - label classification: domain, and which fields are present
  - span extraction
  - normalization into the `{category, params}` contract
- **How to compare.** Against Frozen Fusion, with the same data roles. Test is a final check only.

## 2. Data (`training_bert/data`, read only)

| split | rows | families | composition |
|---|---|---|---|
| train | 17,272 | 4,858 | 6,168 silver (= Frozen Fusion split1 train) + 11,104 synthetic v2 (56 generation batches) |
| val | 3,436 | 1,767 | 2,041 silver (= split1 val) + 1,395 v2 |
| test | 2,181 | 1,664 | = `training_frozen_fusion/data/split1/test.jsonl` (identical rows and labels to `eval/dataset/data/test.jsonl`, plus spans) |

**Preparation, done before this experiment:**
- The v2 rows' ids were renumbered `syn_10657…syn_23155`, so they follow the `syn_` convention.
- The test file was replaced with split1 test, so test numbers compare directly with Frozen Fusion.
- **Leakage checks:** test shares no ids, queries or families with train or val. No val row was ever seen by
  Frozen Fusion (its train is split1 train + `extra_train`).

**Labels:**
- Schema v2: `normalize_row` renames the range-filter keys. Train and val are already v2. Test is normalized at
  load: 103 rows change.
- 13% of train params have no span. These are mostly `סקטורים` and `תתי_קטגוריות`, implied by the item word.
- Train/val spans are all `explicit` and never overlap. Test also has `derived` spans that overlap; span metrics
  count explicit spans only.
- 52 train rows label a span `נפח_אחסון` while the param is `אחסון_GB`. Those spans are skipped, and the data files
  are untouched.

**Split roles:**

| split | role | rows |
|---|---|---|
| train | fit | 17,272 |
| val_eval | every choice: LR, epochs, decoding, normalizer rules | 1,617 |
| val_calib | gate threshold + audit; a check that never drove a choice | 1,819 |
| test | final check, once | 2,181 |

- val_eval and val_calib are split by family (`artifacts/val_split.json`).
- Silver rows keep their Frozen Fusion side.
- v2 families (generation batches of about 200 rows) are split 50/50, seed 42.

## 3. Model (`training_bert/common`)
- **Encoder:** `dicta-il/dictabert`, BERT-base, with a 128k Hebrew vocabulary and a fast tokenizer with offsets.
- **Heads on one encoder:**
  - domain: 4-way, on [CLS]
  - field presence: 69-way sigmoid, on [CLS]
  - BIO span tags: 139 tags, masked to the predicted domain's fields
  - span operator: 7-way, on [mean; first token] of the span
  - span-less values: implied values for `סקטורים`, `תתי_קטגוריות` and `יצרנים`, on [CLS]
- **Loss:** domain CE + 10 × presence BCE + tag CE + op CE + implied CE.
- **Training:**
  - AdamW, weight decay 0.01
  - heads at 10× the encoder LR
  - linear schedule with 6% warmup
  - batch 32
  - bf16 autocast, fp32 master weights
  - gradient clip 1.0
  - max 128 tokens; the longest query is 95
- **BIO alignment:** 97.3% of train spans and 98.1% of val_eval spans round-trip exactly. Almost every miss is a
  one-letter prefix (ו/ש/ל/ב) glued to the token. Overlap recall is 98%.
- **Normalization:** deterministic, reusing `src/extraction`; the runtime code is unchanged. The lexicon is rebuilt
  from this train.
  - numbers: `parse_numeric`, with the operator coming from the op head
  - booleans: polarity from the op head
  - categorical and text fields: the FieldExtractor matchers run on the span text
  - two one-sided bounds of one field are merged into a range
  - finally: taxonomy rules
- **Normalization ceiling:** gold spans through the normalizer give **94.1%** val_eval JSON at first, **95.4%** after
  the E6 fixes (§5).

## 4. Infrastructure

| item | value |
|---|---|
| GPU | Vast.ai instance 61039232: RTX 3090 24 GB, US host, $0.179/h GPU + $0.048/h disk |
| image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, torch 2.5.1+cu124, transformers 4.51.3, driver 580 (`runs/environment*.txt`, `runs/setup.log`) |
| speed | about 43 s per training epoch for base (17k rows), about 77 s for large; the per-epoch val_eval scoring adds about 1 min (CPU normalizer) |
| total cost | about **$1** for the whole experiment (credit $25.00 → about $24) |
| access | `~/.ssh/vast_ed25519`, a key just for Vast with no passphrase. The usual `id_rsa` has one, so it can't be used non-interactively. |
| helpers | `training_bert/vast/remote.sh` (push / setup / run / pull / ssh), `vast/setup.sh`, queue scripts in `queues/` |

**Problems hit along the way:**
- SSH dropped a few times, so every job runs detached (`setsid nohup`) and is polled.
- `pkill -f` matched its own SSH command line. The fix is the bracket pattern (`'[e]7_…'`).
- Piping a file through `ssh cat >` left it empty. The fix is `scp`.
- Git Bash `tar` read `C:` as a host name. The fix is `--force-local`.

## 5. Experiments (all on val_eval unless stated; JSON = JSON exact)

### E0 — smoke tests
- **CPU**, 400 rows: loss 13.7 → 8.4.
- **GPU**, 642 rows in 2 s, bf16 confirmed.
- The pipeline runs end to end.

### Frozen Fusion on the same val_eval (`frozen_fusion_baseline.py`)
- **Overall:** **67.8% JSON**, 68.6% field set, 96.2% domain.
- **Silver rows:** 84.8%, consistent with its earlier 83.8%. **v2 rows: 38.9%.**
- **Its gate:** accepts 43.5% at 84.1%.
- It needed a binding workaround: the `main` commit `d79bfa6` changed two docstrings in `src/extraction`, which
  changes the gate's code hash, so `LocalParser.load` refuses. The script tolerates only that `code` binding.

### E2 — encoder learning rate (seed 42, all train, 8 epochs)

| LR | best JSON (epoch) | field set | span F1 |
|---|---|---|---|
| 1e-5 | 84.5% (8) | 89.7% | 0.832 |
| 2e-5 | 87.6% (8) | 92.4% | 0.860 |
| 3e-5 | 88.4% (7) | 93.0% | 0.864 |
| **5e-5** | **89.4% (8)** | 93.6% | 0.872 |
| 8e-5 | 89.2% (8) | 93.8% | 0.877 |

- **Selected: 5e-5.** It ties 8e-5, and both are flat at epochs 6–8.
- The operator head is about 99.5% accurate from epoch 3 on.
- The "filtered" decoding (keep a span only if the presence head agrees) lost in every epoch, so it was dropped.

### E3 — seed variance (5e-5)
- Seeds 42 / 13 / 1337: 89.4% / 89.6% / 88.9%, **mean 89.3% ± 0.3**.
- Differences under about 1 point are noise.

### E4 — learning curve (5e-5; two nested family-grouped chains × two seeds)

| train rows | ~2.1k | ~4.2k | ~8.2k | ~12k | 17.3k |
|---|---|---|---|---|---|
| JSON (mean) | 70.5% | 80.4% | 85.1% | 87.6% | 89.3% |

- **Still rising, but flattening:** +9.9, +4.7, +2.5, +1.7 points per step.
- **Silver only (6.2k rows): 78.6%.** That's below the mixed curve at a similar size, so the v2 synthetic rows help.
- **Noise at small sizes:** the two chains differ by 7 points at 2k, because which families land in the subset
  matters. By 8k they agree within 0.3.

### E6 — final config with saved weights + normalizer fixes
- The config was retrained with `--save` (3 seeds). Training is deterministic: seed 42 reproduced 89.36% exactly.
- **Oracle analysis (gold spans):** the normalizer failed on 5.8% of val_eval rows.
  - 44 numeric errors: number-word typos, self-corrections
  - 25 categorical paraphrases the lexicon misses
  - 11 categorical typos or negations
  - 9 single values turned into lists
  - 6 open-text prefixes
- **Fixes:** decode-time only, in `common/normalize.py`; runtime `src/` untouched.
  - Misspelt number words: final-letter misuse such as «אלפ», «עשרימ»; doubled final letters such as «מיליוןן»;
    edit distance 1 against the number lexicon. A word is only rewritten when it's unknown and next to a number or
    bound word, and frequent train words are protected («שינה» is not «שישה»).
  - Spelt-out storage sizes become powers of two («חמש מאות ג׳יגה» = 512). Digit «500 גיגה» stays 500, as in the
    labels.
  - **Measured on gold spans:** train 70 fixes / 0 regressions; val_eval 19 / 0. The oracle went from 94.1% to
    95.4%.

| E6, 3 seeds (mean ± sd) | val_eval | val_calib |
|---|---|---|
| old normalizer | 89.3% ± 0.3 | 88.6% ± 0.1 |
| new normalizer | 90.2% ± 0.3 | 89.0% ± 0.1 |

### E7 — learned categorical value head (negative)
- **What it is:** a span head over 502 (field, value) pairs, the values of 34 categorical fields. Every val_eval
  categorical value occurs in train.
- **Result:** seed 42, epoch 8, val_eval JSON by value policy:

| value policy | val_eval JSON |
|---|---|
| lexicon | 89.4% |
| head | 88.2% |
| hybrid | 88.9% |
| confident | 88.9% |
| *E6 seed 42, same normalizer, no head* | *90.4%* |

- The head picks values worse than the lexicon, and training it cost about 1 point overall: the extra objective
  competes with tagging. **Dropped.**
- Its queue was stopped after the seed-42 run.

### E8 — DictaBERT-large (negative)
- **Model:** `dicta-il/dictabert-large`, 24 layers, 1024 hidden, same tokenizer.
- **Result:** seed 42, 8 epochs: lr 3e-5 **89.5%**, lr 2e-5 **88.9%**, against the base single-model 90.4% under the
  same normalizer.
- It starts faster but levels off lower, at about 1.8× the epoch time. **Dropped:** capacity isn't the limit.

### E9 — hybrid decoding with Frozen Fusion's extractor as a fallback
- **Why: error analysis** of E6 seed 42 (`runs/e6_lr5e-5_s42/errors_norm2_val_eval.json`).
  - **Where rows first fail:** 1.3% domain, 4.8% field set, 3.5% values.
  - **69 missed fields:**
    - 28: a span was found but its text gave no value (partial «דול» of «דול פין», glued «מאובזרתבמלוא», «X»)
    - 8: no span, but the presence head knew (≥ 0.5)
    - 33: missed by both; heavy typos such as «בחdרה», «סמס.ונג»
- **The fallback:**
  - `span`: when a span gives no value, the field-local extractor runs on the query, and the value must lie within
    12 chars of the span.
  - `presence@τ`: additionally, for non-numeric fields the presence head predicts (≥ τ) with no span, the extractor
    runs on the whole query.

| E6, 3 seeds (mean ± sd) | val_eval | val_calib |
|---|---|---|
| new normalizer, no fallback | 90.2% ± 0.3 | 89.0% ± 0.1 |
| + fallback `span` | 90.6% ± 0.2 | 89.3% ± 0.2 |
| + fallback `presence@0.5` | 90.7% ± 0.1 | 89.5% ± 0.4 |

### E10 — 3-seed ensemble
- **Method:** average every head's probabilities (domain, presence, tags, span operators, implied values) over the
  3 E6 models.
- **Check:** an ensemble of one model copied twice reproduces the single model exactly.

| ensemble | val_eval | val_calib |
|---|---|---|
| no fallback | 90.9% | 89.6% |
| + `span` | 91.2% | 89.9% |
| + `presence@0.3` | 91.3% | 90.2% |
| **+ `presence@0.5` (chosen)** | **91.2%** | **90.0%** |
| + `presence@0.7` | 91.2% | 89.9% |

- 0.5 was kept, not 0.3: the difference is within noise, and picking the maximum would over-tune.

### E11 — final configuration, test scored once
**Config:** DictaBERT-base, lr 5e-5, 8 epochs, ensemble of seeds 42/13/1337, new normalizer, fallback `presence@0.5`.
All of it was fixed on val_eval before test was read.

| metric | val_eval | val_calib | **test** | Frozen Fusion test |
|---|---|---|---|---|
| JSON exact | 91.2% | 90.0% | **84.8%** | 75.4% |
| field set exact | 94.4% | 93.3% | 91.2% | 76.3% |
| domain | 98.6% | 98.5% | 99.3% | 98.5% |
| span F1 (explicit) | 0.879 | 0.849 | 0.808 | – |

- **Where test rows first fail:** domain 0.5%, field set 8.0%, values 6.7%.
- **Test slices:**
  - silver 87.4%, **gold 79.5%**
  - standard 88.5%, slang 87.6%, **typo 78.0%**
  - second-hand 86.1%, cars 85.4%, **real estate 80.5%**
  - non-search 99.0%
- **Why test is lower than val:** test has no v2 rows, a third of it is the hand-written gold set, and it uses the
  older label conventions. Frozen Fusion showed the same kind of drop (83.8% → 75.4%). The like-for-like comparison
  is +9.4 points on identical test rows.

### E12 — confidence gate for LLM routing (`gate.py`)
- **Signals,** added to `predict`:
  - domain probability
  - lowest per-token tag probability
  - lowest operator probability
  - ensemble agreement: members that decode the same domain and spans
  - presence margins, field and span counts, whether the fallback was used, query length, domain
- **Gate:** logistic regression fitted on val_eval rows (exact or not).
- **Threshold:** the largest coverage on val_calib half A (by family) whose one-sided 95% Wilson lower bound reaches
  target + 0.01.
- **Audit:** val_calib half B. **Test:** checked once.

| target | select: share answered locally / accuracy (lower bound) | audit: share / accuracy (lower bound) | test: share / accuracy |
|---|---|---|---|
| 0.90 | 97.9% / 93.2% (0.911) | 97.2% / 91.7% (0.903) | 95.9% / 87.1% |
| 0.92 | 93.2% / 95.0% (0.931) | 94.1% / 93.2% (0.919) | 91.5% / 89.1% |
| 0.94 | 86.9% / 96.9% (0.952) | 88.2% / 94.9% (0.937) | 83.7% / 91.5% |
| **0.95** | 81.5% / 97.6% (0.960) | 84.2% / 95.1% (**0.939**) | **77.2% / 92.4%** |
| 0.96 | 69.4% / 98.6% (0.971) | 74.3% / 95.7% (0.945) | 66.3% / 93.5% |
| 0.97+ | not reachable | | |

- **At 95%:** the audit measures 95.1%, but its lower bound (0.939) misses the target, so 95% is not certified. On
  test the accepted parses are 92.4% accurate, another sign of the val → test shift.
- **Cost reading,** using the user's figures (40% of traffic = 13.2M searches = $1,300/month, so ~33M searches a
  month at ~$0.10 per 1k LLM calls):
  - The 0.95 gate on test sends 22.8% to the LLM: ~7.5M calls, **~$740/month**, against ~$3,250 for LLM-only.
  - The GPT-6 Luna parser reportedly reaches 89.36% on test (not measured here). If Luna keeps that rate on the
    rejected queries, the routed system's test accuracy is about 0.772 × 92.4% + 0.228 × 89.4% ≈ 91.7%.
  - The rejected queries are the hard ones, so the true figure needs Luna's per-row outputs.

## 6. Conclusions
1. **Fine-tuning works:** +9.4 test points over Frozen Fusion, and +23 points on the same val_eval rows.
2. **The limit is data, not capacity.** The large encoder and the value head both failed. The learning curve still
   gains about 1 point per 5k rows. val → test is a 6-point gap: gold, typo and real-estate rows score lowest.
3. **Decoding is cheap and real:** the normalizer fixes (+1.0), the ensemble (+0.6) and the extractor fallback
   (+0.3–0.6) all held on val_calib.
4. **95% local accuracy is not certified with val-based calibration.** About 92% on test at 77% coverage is what the
   current gate delivers.

## 7. Open issues / next
- **`main`:** the gate's code hash is broken (commit `d79bfa6`); with the extractor enabled, the API won't load the
  local parser. If the tagger replaces Frozen Fusion, it gets its own gate artifact anyway.
- **Label noise** found along the way:
  - «עם חניה» is labelled {min:1} in some rows and {min:2} in others
  - span ops sometimes disagree with the value shape (e.g. «משנת 2024» exact against {min})
  - the `נפח_אחסון` / `אחסון_GB` span-key mismatch
- **To compare with Luna:** its per-row test outputs, for a measured routed accuracy and an agreement signal for the
  gate.
- **Weights:** the 3 final checkpoints are in `runs/e6_lr5e-5_s*/model.pt` (git-ignored; SHA-256 in
  `runs/weights_sha256.txt`). The E7 and E8 checkpoints exist only on the Vast box.

## 8. Reproduce

```bash
# local: data layer + normalizer checks
pytest tests/test_bert_data.py
python -m training_bert.frozen_fusion_baseline                      # Frozen Fusion on val_eval (needs models/multilingual-e5-small)

# on the GPU box (training_bert/vast/remote.sh push; setup; then run a queue from experiment1/queues)
python -m training_bert.train --out <run> --lr 5e-5 --seed 42 --epochs 8 --save        # one E6 member (~16 min)
python -m training_bert.final_test --tag=norm2 <run> [<run> ...]                         # re-score val_eval / val_calib
python -m training_bert.rescore_decode <run> --tag=norm2 --split=val_eval               # fallback variants (CPU)
python -m training_bert.final_test --ensemble=<out> --tag=final --fallback=presence --with-test <3 runs>   # E11
python -m training_bert.final_test --ensemble=<out> --tag=gate --fallback=presence --with-test <3 runs>    # E12 inputs
python -m training_bert.gate <out> --target 0.95                                         # E12
python -m training_bert.experiment1.summarize                                           # results/summary.json
```

**Files:**
- `runs/<run>/metrics.json`: config + per-epoch val_eval metrics
- `final*.json`, `predictions*_<split>.jsonl`: per-row records with raw model output
- `decode*.json`: fallback variants
- `errors*.json`: error analysis
- `queues/`: the scripts exactly as they ran, with paths as they were (`training_bert/runs`)

**Run names:**
- `e2_*` LR sweep, `e3_*` seeds, `e4_*` learning curve
- `e6_*` final config (saved weights)
- `e7_*` value head, `e8_*` large
- `e10_*` ensemble, `e11_*` final test, `e12_*` gate
