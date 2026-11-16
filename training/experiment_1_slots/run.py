"""Exp1-slots: frozen E5 embeddings + logistic-regression heads on 70 merged point/range slots.

    python training/experiment_1_slots/run.py --sweep --curve --final
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.runner import Spec, cli  # noqa: E402

SPEC = Spec("exp1-slots", "e5", "slots", Path(__file__).resolve().parent,
            "E5 (frozen, 384-d) + LR heads, 70 slots")

if __name__ == "__main__":
    cli(SPEC)
