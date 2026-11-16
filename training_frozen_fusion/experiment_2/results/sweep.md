# exp2 sweep (val_eval, development)

| C | w | domain_accuracy | slot_micro_f1 | slot_macro_f1 | slot_exact_set | slot_exact_set_rules | domain_ece | strict_exact_deployable |
|---|---|---|---|---|---|---|---|---|
| 1 | None | 0.979 | 0.907 | 0.832 | 0.637 | 0.637 | 0.024 | 0.608 |
| 4 | None | 0.983 | 0.935 | 0.876 | 0.724 | 0.725 | 0.009 | 0.690 |
| 16 | None | 0.982 | 0.944 | 0.903 | 0.767 | 0.767 | 0.011 | 0.731 |
| 64 | None | 0.984 | 0.945 | 0.907 | 0.764 | 0.764 | 0.011 | 0.729 |
| 256 | None | 0.983 | 0.942 | 0.895 | 0.745 | 0.746 | 0.007 | 0.711 |

Selected: C=16 w=None (max slot exact-set, tie -> slot micro-F1)