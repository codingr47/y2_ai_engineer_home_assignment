# Exp7: Exp3 v2 retrained with the extra training batch

**Setup.**
- Model: Exp3 v2 (E5 + TF-IDF early fusion, LR heads, w=1).
- Training data: split1 train (6,171) plus `training/data/extra_train` (3,188 silver rows, 2,132 units; 90% of them
  with labels not in the old pool). Labels are converted to v2 at load time.
- Checks run here: the hash matches the manifest, and there is no exact or near-duplicate overlap with split1 or
  the routing batch.
- Data roles:
  - val_calib: calibration and thresholds.
  - val_eval: selection.
  - routing dev (fit + selection): development.
  - routing audit: sealed.
  - test: not scored.
- Extractor: the current one, after the experiment_5 fixes, the same for every row of the table.

| model | train rows | val_eval JSON | val_eval field set | routing JSON | routing field set | routing typo JSON |
|---|---|---|---|---|---|---|
| shipped Exp3 v2 | 6,171 | 81.7% | 79.4% | 75.4% | 77.3% | 66.1% |
| + 50% extra, C=16 | 7,766 | 82.1% | 78.5% | 77.5% | 78.9% | 69.4% |
| **+ 100% extra, C=16** | 9,359 | **83.8%** | **82.2%** | **77.4%** | **80.7%** | 69.3% |
| + 100% extra, C=64 | 9,359 | 83.9% | 80.9% | 77.1% | 80.3% | 68.1% |

**Selected: + 100% extra, C=16.** The existing rule picks the best field-set exact on val_eval; JSON is tied with
C=64.
- Gain over the shipped classifier: about +2 JSON points on both distributions, and +2.8 / +3.4 field-set points.
- The routing JSON gain is flat from 50% to 100% of the batch. Returns are diminishing: another batch of this size
  would probably add about 1 point, not the 15+ points the 95% gate target needs.
- These are development numbers. No paired bootstrap was run.

```bash
python training/experiment_7_extra_train/run.py     # ~4 min on CPU; artifacts/ (models, embeddings) git-ignored
```
