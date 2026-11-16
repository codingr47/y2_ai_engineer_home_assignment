"""Exp3: early fusion hstack([w * E5, TF-IDF]) + logistic-regression heads on 70 merged slots.
C and w are tuned jointly because block scale interacts with L2 regularisation.

    python training/experiment_3/run.py --sweep --curve --final
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.runner import W_GRID, Spec, cli  # noqa: E402

SPEC = Spec("exp3", "e5+tfidf", "slots", Path(__file__).resolve().parent,
            "early fusion [w*E5, TF-IDF] + LR heads, 70 slots")

if __name__ == "__main__":
    cli(SPEC, w_grid=W_GRID)
