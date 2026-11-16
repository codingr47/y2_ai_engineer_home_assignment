"""Re-export: the span-local operator detector now lives in src/extraction/operators.py (single source)."""
from . import paths  # noqa: F401  (puts the repo root on sys.path)

from src.extraction.operators import (  # noqa: E402,F401
    EXACT, MAX, MIN, MIN_NEGATED, OPERATORS, RANGE, detect_operator, is_bounded)
