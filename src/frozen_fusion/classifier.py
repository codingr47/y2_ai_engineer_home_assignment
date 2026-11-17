"""Portable inference for exported Exp3 fusion artifacts. No training imports."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from scipy.special import expit, softmax
from sklearn.preprocessing import normalize


@dataclass
class Classification:
    domain: str
    fields: list[str]
    domain_probs: np.ndarray
    field_probs: np.ndarray
    selected: np.ndarray
    thresholds: np.ndarray
    in_domain: np.ndarray


class FusionClassifier:
    def __init__(self, artifact: dict, encoder):
        if artifact.get("schema") != "exp3-v2-runtime-1":
            raise ValueError("unsupported classifier artifact")
        self.artifact = artifact
        self.encoder = encoder

    @classmethod
    def load(cls, path: Path, encoder_path: Path):
        if not encoder_path.is_dir():
            raise ValueError("E5_MODEL_PATH must point to a provisioned local encoder")
        from sentence_transformers import SentenceTransformer
        artifact = joblib.load(path)
        encoder = SentenceTransformer(str(encoder_path), local_files_only=True)
        encoder.eval()
        return cls(artifact, encoder)

    def features(self, queries):
        a = self.artifact
        embeddings = self.encoder.encode([a["query_prefix"] + q for q in queries],
                                         normalize_embeddings=True, show_progress_bar=False,
                                         convert_to_numpy=True).astype(np.float32)
        if embeddings.shape[1] != a["embedding_dimension"]:
            raise ValueError("encoder dimension mismatch")
        tf = normalize(sparse.hstack([a["word"].transform(queries), a["char"].transform(queries)], format="csr"))
        return sparse.hstack([sparse.csr_matrix(a["weight"] * embeddings), tf], format="csr")

    def predict_X(self, X):
        a = self.artifact
        if a["domain_method"] == "temperature":
            pd = softmax(a["domain_model"].decision_function(X) / a["temperature"], axis=1)
        else:
            pd = a["domain_calibrator"].predict_proba(X)
        raw = np.clip(a["field_model"].predict_proba(X), 1e-7, 1 - 1e-7)
        pf = expit(a["platt_a"] * np.log(raw / (1 - raw)) + a["platt_b"])
        domains = a["classes"][pd.argmax(axis=1)]
        mask = np.array([[f.split("/", 1)[0] == d for f in a["vocab"]] for d in domains])
        pf *= mask
        return pd, pf, pf >= a["thresholds"]

    def classify(self, query: str) -> Classification:
        pd, pf, selected = self.predict_X(self.features([query]))
        a = self.artifact
        domain = str(a["classes"][pd[0].argmax()])
        return Classification(domain, [f for f, yes in zip(a["vocab"], selected[0]) if yes],
                              pd[0], pf[0], selected[0], a["thresholds"],
                              np.array([f.startswith(domain + "/") for f in a["vocab"]]))
