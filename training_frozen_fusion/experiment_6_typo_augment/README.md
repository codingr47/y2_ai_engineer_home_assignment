# Exp6: typo augmentation of train (negative result)

**Question.** The parser is weakest on typo queries in the fresh routing batch (64% exact-JSON). Does adding noisy
copies of the train queries, with the same labels, improve the fusion classifier's field recall?

**Setup.** This is Exp3 v2 with the frozen config (E5 + TF-IDF early fusion, LR heads, C=16, w=1).
- **Augmentation:** each train row gets k noisy copies. A copy makes 1–2 edits to Hebrew words of 3+ letters
  (deletion, doubling, adjacent swap, confusable letters ט/ת כ/ק ס/ש א/ע ו/ב, glued or split words). Digits and
  Latin are never touched, so the labels stay valid.
- **Data roles:** val_calib calibrates; val_eval and routing dev (fit + selection) are development; the routing
  audit is sealed; test is not used.
- **Extractor:** these results use the extractor *before* the experiment_5 fixes, the same for every k.

| model | val_eval JSON | val_eval field set | routing dev JSON | routing field set | routing typo JSON |
|---|---|---|---|---|---|
| shipped Exp3 v2 | 80.6% | 79.4% | 73.7% | 77.3% | 64.4% |
| k = 0 (rebuild) | 80.6% | 79.4% | 73.7% | 77.3% | 64.4% |
| k = 1 | 80.3% | 78.6% | 73.7% | 75.9% | 64.7% |
| k = 2 | 80.4% | 78.7% | 73.5% | 76.3% | 65.6% |

- k = 0 reproduces the shipped classifier exactly, which validates the pipeline.
- Augmentation buys at most about 1 point on typo rows, loses it on standard and slang, and lowers field-set
  accuracy. **Not adopted.** The typos in the generated data, such as phonetic spellings («ליסינכ»), merged words
  and missing letters in key words, are not captured by random edits, and the classifier's misses are mostly new
  phrasings rather than typos.

```bash
python training_frozen_fusion/experiment_6_typo_augment/run.py --k 0 1 2    # ~2.5 min on CPU; artifacts/ is git-ignored
```
