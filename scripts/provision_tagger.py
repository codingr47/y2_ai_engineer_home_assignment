"""Provision the tagger parser's weights and encoder files into models/fine-tuned-tagger (git-ignored).

    python -m scripts.provision_tagger [--source training/training_bert/experiment1/runs]

- Copies each ensemble member listed in src/artifacts/tagger/members.json from `<source>/<run>/model.pt` to
  `models/fine-tuned-tagger/<file>`, refusing any file whose SHA-256 differs from members.json.
- Saves the encoder's config and tokenizer (no weights: the fine-tuned checkpoints carry them) to
  `models/fine-tuned-tagger/encoder`.
"""
import argparse
import json
import shutil
from pathlib import Path

from src.config.settings import Settings
from src.tagger.parser import sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=ROOT / "training/training_bert/experiment1/runs")
    a = ap.parse_args()
    s = Settings()
    spec = json.loads((s.tagger_artifact_dir / "members.json").read_text(encoding="utf-8"))
    s.tagger_model_dir.mkdir(parents=True, exist_ok=True)
    for m in spec["members"]:
        dst = s.tagger_model_dir / m["file"]
        if dst.exists() and sha256(dst) == m["sha256"]:
            print(f"{m['file']}: present")
            continue
        src = a.source / m["run"] / "model.pt"
        if sha256(src) != m["sha256"]:
            raise SystemExit(f"{src} does not match members.json")
        shutil.copyfile(src, dst)
        print(f"{m['file']}: copied from {src}")
    from transformers import AutoConfig, AutoTokenizer
    enc = s.tagger_model_dir / "encoder"
    AutoConfig.from_pretrained(spec["encoder"]).save_pretrained(enc)
    AutoTokenizer.from_pretrained(spec["encoder"]).save_pretrained(enc)
    print(f"encoder config + tokenizer: {enc}")


if __name__ == "__main__":
    main()
