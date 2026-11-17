"""Paths and constants for experiment 1 (frozen E5 embeddings + logistic regression heads)."""
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR.parent / "data" / "split1"
ARTIFACTS = EXP_DIR / "artifacts"
LABELS_DIR = ARTIFACTS / "labels"
EMB_DIR = ARTIFACTS / "embeddings"
MODELS_DIR = ARTIFACTS / "models"
RESULTS_DIR = ARTIFACTS / "results"

MODEL_NAME = "intfloat/multilingual-e5-small"
EMB_DIM = 384
QUERY_PREFIX = "query: "
BATCH_SIZE = 32

SEED = 0
NONE_DOMAIN = "none"  # domain label for non-search rows (domain == null)
SPLITS = ("train", "val_calib", "val_eval", "test")

TRAIN_SIZES = (250, 500, 1000, 2000, 4000, None)  # None = full train
CURVE_SEEDS = (0, 1, 2)
