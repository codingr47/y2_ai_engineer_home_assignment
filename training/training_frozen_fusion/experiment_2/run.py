"""Exp2: TF-IDF word 1-2-grams + char_wb 2-5-grams + logistic-regression heads on 70 merged slots.

    python training/training_frozen_fusion/experiment_2/run.py --sweep --curve --final
    python training/training_frozen_fusion/experiment_2/run.py --v2 --final-test   # schema v2, frozen config
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.runner import Spec, cli  # noqa: E402

SPEC = Spec("exp2", "tfidf", "slots", Path(__file__).resolve().parent,
            "TF-IDF (word 1-2 + char_wb 2-5) + LR heads, 70 slots")

# Label schema v2 (training/training_frozen_fusion/shared/normalize.py): same features and frozen config, labels normalized at load
# time (טווח_* and vehicle שנות_ייצור folded into point fields; data files untouched) -> results_v2/
SPEC_V2 = Spec("exp2-v2", SPEC.kind, "v2", SPEC.out_dir, SPEC.description.replace("70 slots", "69 v2 fields"),
               results_name="results_v2", selected_from=SPEC.results_dir / "selected.json")

if __name__ == "__main__":
    cli(SPEC, spec_v2=SPEC_V2)
