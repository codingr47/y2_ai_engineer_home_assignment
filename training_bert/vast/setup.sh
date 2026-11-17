#!/usr/bin/env bash
# On the box: dependencies, encoder cache, GPU / bf16 check.
set -euo pipefail
pip install -q -r training_bert/requirements.txt
python - <<'PY'
import torch
from transformers import AutoModel, AutoTokenizer
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0),
      "bf16", torch.cuda.is_bf16_supported())
AutoTokenizer.from_pretrained("dicta-il/dictabert"); AutoModel.from_pretrained("dicta-il/dictabert")
print("encoder cached")
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
