# Exp3 — early fusion [w·E5, TF-IDF] + LR heads on 70 slots

Both blocks have unit row norm; C and w were tuned jointly (scale interacts with L2). Shared code: `training/shared/` (runner, features, slots, taxonomy rules). Joint analysis:
`training/comparison/REPORT.md`.

```bash
python training/experiment_3/run.py --sweep        # val_eval selects the config -> results/selected.json
python training/experiment_3/run.py --curve        # learning curve at the selected config
python training/experiment_3/run.py --final-test   # frozen config: val_calib/val_eval/test predictions + final_test.json
```

Selected on val_eval (development): **C=16, w=1**.

| split | domain acc | slot micro-F1 | slot macro-F1 | slot exact-set | strict exact-set (deployable) |
|---|---|---|---|---|---|
| val_eval (development) | 0.987 | 0.952 | 0.918 | 0.798 | 0.760 |
| test (frozen) | 0.984 | 0.930 | 0.897 | 0.733 | 0.709 |

Results: `results/sweep.md`, `results/learning_curve.json`, `results/final_test.json`. The per-query prediction
records (`results/predictions_*.jsonl`: domain/slot probabilities, selected/rejected-probability margins, and
`slot_exact_match` as an evaluation target only) are git-ignored; regenerate them with `--final-test`.

**Label schema v2.** `python training/experiment_3/run.py --v2 --final-test` refits the same frozen config on labels
normalized at load time (`training/shared/normalize.py`; data files untouched). It writes `results_v2/` and
`artifacts/models_v2/`. See `training/comparison/REPORT.md` §9.
