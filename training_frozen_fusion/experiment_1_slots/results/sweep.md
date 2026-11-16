# exp1-slots sweep (val_eval, development)

| C | w | domain_accuracy | slot_micro_f1 | slot_macro_f1 | slot_exact_set | slot_exact_set_rules | domain_ece | strict_exact_deployable |
|---|---|---|---|---|---|---|---|---|
| 1 | None | 0.945 | 0.811 | 0.715 | 0.421 | 0.422 | 0.052 | 0.403 |
| 4 | None | 0.964 | 0.863 | 0.791 | 0.528 | 0.532 | 0.018 | 0.500 |
| 16 | None | 0.966 | 0.885 | 0.834 | 0.591 | 0.591 | 0.014 | 0.563 |
| 64 | None | 0.961 | 0.898 | 0.867 | 0.646 | 0.648 | 0.015 | 0.611 |
| 256 | None | 0.961 | 0.892 | 0.860 | 0.615 | 0.615 | 0.014 | 0.581 |

Selected: C=64 w=None (max slot exact-set, tie -> slot micro-F1)