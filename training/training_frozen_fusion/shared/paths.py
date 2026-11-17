"""Puts experiment_1's modules (train, evaluate, embed, config, run_experiment) on sys.path.

They are imported by bare name because experiment_1/artifacts/models/heads.joblib unpickles from module `train`;
moving them would break that artifact.
"""
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
EXP1 = TRAINING / "experiment_1"
REPO = TRAINING.parent.parent
DATA_DIR = TRAINING / "data" / "split1"
TAXONOMY = REPO / "yad2_search_taxonomy.json"

for p in (str(TRAINING), str(EXP1), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
