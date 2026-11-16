# Exp1-slots — frozen E5 + LR heads on 70 merged slots

Same features and heads as Experiment 1; only the label space changes (each point/range twin pair → one slot). Shared code: `training/shared/` (runner, features, slots, taxonomy rules). Joint analysis:
`training/comparison/REPORT.md`.

```bash
python training/experiment_1_slots/run.py --sweep        # val_eval selects the config -> results/selected.json
python training/experiment_1_slots/run.py --curve        # learning curve at the selected config
python training/experiment_1_slots/run.py --final-test   # frozen config: val_calib/val_eval/test predictions + final_test.json
```

Selected on val_eval (development): **C=64**.

| split | domain acc | slot micro-F1 | slot macro-F1 | slot exact-set | strict exact-set (deployable) |
|---|---|---|---|---|---|
| val_eval (development) | 0.961 | 0.898 | 0.867 | 0.646 | 0.611 |
| test (frozen) | 0.953 | 0.865 | 0.823 | 0.557 | 0.539 |

Results: `results/sweep.md`, `results/learning_curve.json`, `results/final_test.json`. The per-query prediction
records (`results/predictions_*.jsonl`: domain/slot probabilities, selected/rejected-probability margins, and
`slot_exact_match` as an evaluation target only) are git-ignored; regenerate them with `--final-test`.
