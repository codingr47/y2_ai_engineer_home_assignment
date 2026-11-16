"""Exp3: early fusion hstack([w * E5, TF-IDF]) + logistic-regression heads on 70 merged slots.
C and w are tuned jointly because block scale interacts with L2 regularisation.

    python training/experiment_3/run.py --sweep --curve --final
    python training/experiment_3/run.py --v2 --final-test   # schema v2, frozen config
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.runner import W_GRID, Spec, cli  # noqa: E402

SPEC = Spec("exp3", "e5+tfidf", "slots", Path(__file__).resolve().parent,
            "early fusion [w*E5, TF-IDF] + LR heads, 70 slots")

# Label schema v2 (training/shared/normalize.py): same features and frozen config, labels normalized at load
# time (טווח_* and vehicle שנות_ייצור folded into point fields; data files untouched) -> results_v2/
SPEC_V2 = Spec("exp3-v2", SPEC.kind, "v2", SPEC.out_dir, SPEC.description.replace("70 slots", "69 v2 fields"),
               results_name="results_v2", selected_from=SPEC.results_dir / "selected.json")

if __name__ == "__main__":
    cli(SPEC, w_grid=W_GRID, spec_v2=SPEC_V2)
