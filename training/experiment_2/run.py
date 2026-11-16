"""Exp2: TF-IDF word 1-2-grams + char_wb 2-5-grams + logistic-regression heads on 70 merged slots.

    python training/experiment_2/run.py --sweep --curve --final
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.runner import Spec, cli  # noqa: E402

SPEC = Spec("exp2", "tfidf", "slots", Path(__file__).resolve().parent,
            "TF-IDF (word 1-2 + char_wb 2-5) + LR heads, 70 slots")

if __name__ == "__main__":
    cli(SPEC)
