"""Encode every split once with the frozen E5 encoder and cache the vectors.

Cache: artifacts/embeddings/<split>.npy (float32, L2-normalised, row order = labels/<split>.json)
       artifacts/embeddings/<split>.meta.json  {model, prefix, n, ids_sha}
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from config import BATCH_SIZE, EMB_DIM, EMB_DIR, LABELS_DIR, MODEL_NAME, QUERY_PREFIX, SPLITS

_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(MODEL_NAME)
        _encoder.eval()
        for p in _encoder.parameters():  # frozen: never trained
            p.requires_grad_(False)
    return _encoder


def encode(queries: list[str]) -> np.ndarray:
    return get_encoder().encode(
        [QUERY_PREFIX + q for q in queries],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def _ids_sha(ids):
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]


def load_or_encode(split: str) -> np.ndarray:
    meta_rows = json.loads((LABELS_DIR / f"{split}_meta.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in meta_rows]
    want = {"model": MODEL_NAME, "prefix": QUERY_PREFIX, "n": len(ids), "ids_sha": _ids_sha(ids)}
    npy, meta = EMB_DIR / f"{split}.npy", EMB_DIR / f"{split}.meta.json"
    if npy.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == want:
        print(f"[cache] {split}")
        return np.load(npy)
    print(f"[encode] {split} ({len(ids)} rows)")
    X = encode([r["query"] for r in meta_rows])
    assert X.shape == (len(ids), EMB_DIM)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(npy, X)
    meta.write_text(json.dumps(want), encoding="utf-8")
    return X


def main():
    for s in SPLITS:
        X = load_or_encode(s)
        norms = np.linalg.norm(X, axis=1)
        print(f"{s:10s} shape={X.shape} norm[min,max]=[{norms.min():.4f},{norms.max():.4f}]")


if __name__ == "__main__":
    main()
