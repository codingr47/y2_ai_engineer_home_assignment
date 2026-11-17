set -euo pipefail
python -m training.training_bert.train --out training/training_bert/runs/e0_smoke --train-size 500 --eval-size 300 --epochs 1
