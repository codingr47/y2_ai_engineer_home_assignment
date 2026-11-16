"""Feature extractors. Each is fit on a run's training subset only and transforms a Split (cached E5 vectors and
query text are both carried by the Split) or raw texts (for end-to-end latency).

Both blocks have unit L2 row norm, so in ConcatFeatures the weight w is the only relative scale between them.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import paths  # noqa: F401  (puts experiment_1 on sys.path)

# Hebrew letters + digits + Latin, keeping geresh/gershayim and in-word quote marks (ק״מ, מ"ר, קוטג׳)
TOKEN_PATTERN = r"(?u)[\w״׳\"']+"


class E5Features:
    name = "e5"

    def fit(self, split):
        return self

    def transform(self, split):
        return np.asarray(split.X, dtype=np.float32)  # cached, already L2-normalised

    def transform_texts(self, texts, batch_size: int = 32):
        # same encoder, prefix and normalisation as experiment_1/embed.py, without the progress bar
        from config import QUERY_PREFIX
        from embed import get_encoder

        return get_encoder().encode([QUERY_PREFIX + t for t in texts], batch_size=batch_size,
                                    normalize_embeddings=True, show_progress_bar=False,
                                    convert_to_numpy=True).astype(np.float32)


class TfidfFeatures:
    name = "tfidf"

    def __init__(self, min_df: int = 2):
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=TOKEN_PATTERN,
                                    sublinear_tf=True, min_df=min_df, lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=min_df,
                                    lowercase=True)

    def fit(self, split):
        texts = [m["query"] for m in split.meta]
        self.word.fit(texts)
        self.char.fit(texts)
        return self

    def transform_texts(self, texts):
        texts = list(texts)
        X = sparse.hstack([self.word.transform(texts), self.char.transform(texts)], format="csr")
        return normalize(X)  # one unit-norm block

    def transform(self, split):
        return self.transform_texts([m["query"] for m in split.meta])

    @property
    def n_features(self):
        return len(self.word.vocabulary_) + len(self.char.vocabulary_)


class ConcatFeatures:
    """Early fusion: hstack([w * E5, TF-IDF]); C and w are tuned jointly (scale interacts with L2)."""

    name = "e5+tfidf"

    def __init__(self, w: float = 1.0, min_df: int = 2):
        self.w = w
        self.e5 = E5Features()
        self.tfidf = TfidfFeatures(min_df=min_df)

    def fit(self, split):
        self.tfidf.fit(split)
        return self

    def _join(self, e5, tf):
        return sparse.hstack([sparse.csr_matrix(self.w * e5), tf], format="csr")

    def transform(self, split):
        return self._join(self.e5.transform(split), self.tfidf.transform(split))

    def transform_texts(self, texts):
        texts = list(texts)
        return self._join(self.e5.transform_texts(texts), self.tfidf.transform_texts(texts))


def make_featurizer(kind: str, w: float = 1.0):
    return {"e5": E5Features, "tfidf": TfidfFeatures}[kind]() if kind != "e5+tfidf" else ConcatFeatures(w=w)
