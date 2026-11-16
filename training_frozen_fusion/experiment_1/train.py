"""Domain + multi-label field heads on frozen embeddings, calibrated on val_calib."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from config import LABELS_DIR, NONE_DOMAIN, SEED
from embed import load_or_encode

MIN_PLATT = 5        # positives (and negatives) in val_calib needed to Platt-calibrate a field
MIN_PER_FIELD_THR = 10  # positives in val_calib needed for a per-field threshold


# ---------------------------------------------------------------- data
@dataclass
class Split:
    name: str
    X: np.ndarray
    labels: list[dict]
    meta: list[dict]

    @property
    def domains(self):
        return np.array([r["domain_label"] for r in self.labels])

    @property
    def fields(self):
        return [r["field_labels"] for r in self.labels]

    def subset(self, idx):
        idx = np.asarray(idx)
        return Split(self.name, self.X[idx], [self.labels[i] for i in idx], [self.meta[i] for i in idx])


def load_split(name: str) -> Split:
    labels = json.loads((LABELS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    meta = json.loads((LABELS_DIR / f"{name}_meta.json").read_text(encoding="utf-8"))
    X = load_or_encode(name)
    assert len(labels) == len(meta) == len(X)
    return Split(name, X, labels, meta)


def load_vocab() -> list[str]:
    return json.loads((LABELS_DIR / "field_vocab.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- domain calibration
class TemperatureScaler:
    """softmax(logits / T); one parameter, cannot overfit a 1k-row calibration set."""

    def fit(self, logits, y_idx):
        res = minimize_scalar(lambda t: log_loss(y_idx, softmax(logits / t, axis=1), labels=range(logits.shape[1])),
                              bounds=(0.05, 20), method="bounded")
        self.T = float(res.x)
        return self

    def transform(self, logits):
        return softmax(logits / self.T, axis=1)


class DomainCalibrator:
    """Picks temperature / sigmoid / isotonic by 5-fold CV log-loss inside val_calib, refits on all of it."""

    METHODS = ("temperature", "sigmoid", "isotonic")

    def __init__(self, model: LogisticRegression):
        self.model = model
        self.classes_ = model.classes_

    def _fit_one(self, method, X, y):
        if method == "temperature":
            y_idx = np.searchsorted(self.classes_, y)
            return TemperatureScaler().fit(self.model.decision_function(X), y_idx)
        return CalibratedClassifierCV(FrozenEstimator(self.model), method=method).fit(X, y)

    def _apply(self, cal, X):
        if isinstance(cal, TemperatureScaler):
            return cal.transform(self.model.decision_function(X))
        return cal.predict_proba(X)

    def fit(self, X, y):
        y = np.asarray(y)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        self.cv_logloss = {}
        for m in self.METHODS:
            losses = []
            for tr, te in skf.split(X, y):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = self._apply(self._fit_one(m, X[tr], y[tr]), X[te])
                losses.append(log_loss(y[te], np.clip(p, 1e-7, 1), labels=self.classes_))
            self.cv_logloss[m] = float(np.mean(losses))
        self.method = min(self.cv_logloss, key=self.cv_logloss.get)
        self.cal = self._fit_one(self.method, X, y)
        return self

    def predict_proba(self, X):
        return self._apply(self.cal, X)


# ---------------------------------------------------------------- field calibration + thresholds
class FieldCalibrator:
    """Per-field Platt scaling on the OvR logit (fields with enough calib support), then thresholds
    tuned for micro-F1: one global, plus per-field where val_calib has >= MIN_PER_FIELD_THR positives."""

    def __init__(self, model: OneVsRestClassifier):
        self.model = model

    def logits(self, X):
        P = np.clip(self.model.predict_proba(X), 1e-7, 1 - 1e-7)
        return np.log(P / (1 - P))

    def fit(self, X, Y, mask=None):
        Z = self.logits(X)
        k = Y.shape[1]
        self.a, self.b = np.ones(k), np.zeros(k)
        self.platt = np.zeros(k, bool)
        for j in range(k):
            pos = int(Y[:, j].sum())
            if pos >= MIN_PLATT and len(Y) - pos >= MIN_PLATT:
                lr = LogisticRegression(C=1e4, max_iter=1000).fit(Z[:, [j]], Y[:, j])
                self.a[j], self.b[j] = lr.coef_[0, 0], lr.intercept_[0]
                self.platt[j] = True
        P = self.transform(X)
        if mask is not None:
            P = P * mask
        grid = np.round(np.arange(0.05, 0.96, 0.025), 3)
        self.global_thr = float(max(grid, key=lambda t: _micro_f1(Y, P >= t)))
        self.thr = np.full(k, self.global_thr)
        for j in range(k):
            if Y[:, j].sum() >= MIN_PER_FIELD_THR:
                def score(t, j=j):
                    pred = P >= self.thr
                    pred[:, j] = P[:, j] >= t
                    return _micro_f1(Y, pred)
                self.thr[j] = max(grid, key=score)
        return self

    def transform(self, X):
        return expit(self.a * self.logits(X) + self.b)


def _micro_f1(Y, pred):
    tp = np.logical_and(Y == 1, pred).sum()
    fp = np.logical_and(Y == 0, pred).sum()
    fn = np.logical_and(Y == 1, ~pred).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def domain_mask(domains: np.ndarray, vocab: list[str]) -> np.ndarray:
    """1 where the field belongs to the row's domain; all-zero rows for 'none'."""
    field_dom = np.array([v.split("/", 1)[0] for v in vocab])
    return (field_dom[None, :] == np.asarray(domains)[:, None]).astype(float)


# ---------------------------------------------------------------- pipeline
@dataclass
class Heads:
    vocab: list[str]
    C: float = 1.0
    mlb: MultiLabelBinarizer = None
    domain_model: LogisticRegression = None
    field_model: OneVsRestClassifier = None
    domain_cal: DomainCalibrator = None
    field_cal: FieldCalibrator = None
    constant_fields: list[str] = field(default_factory=list)

    def fit(self, train: Split):
        self.mlb = MultiLabelBinarizer(classes=self.vocab)
        Y = self.mlb.fit_transform(train.fields)
        self.constant_fields = [self.vocab[j] for j in range(len(self.vocab)) if Y[:, j].min() == Y[:, j].max()]
        self.domain_model = LogisticRegression(C=self.C, max_iter=2000).fit(train.X, train.domains)
        with warnings.catch_warnings():  # OvR emits one warning per label absent from a small subset
            warnings.simplefilter("ignore", UserWarning)
            self.field_model = OneVsRestClassifier(LogisticRegression(C=self.C, max_iter=2000), n_jobs=-1)
            self.field_model.fit(train.X, Y)
        return self

    def calibrate(self, calib: Split):
        self.domain_cal = DomainCalibrator(self.domain_model).fit(calib.X, calib.domains)
        Y = self.mlb.transform(calib.fields)
        # thresholds are tuned under the same domain mask used at inference
        pred_dom = self.domain_cal.classes_[self.domain_cal.predict_proba(calib.X).argmax(1)]
        self.field_cal = FieldCalibrator(self.field_model).fit(calib.X, Y, mask=domain_mask(pred_dom, self.vocab))
        return self

    # every variant returns (domain_proba, field_proba, field_pred) so evaluation is uniform
    def predict(self, X, variant="full"):
        classes = self.domain_model.classes_
        if variant == "raw":  # uncalibrated, threshold 0.5, no mask
            Pd = self.domain_model.predict_proba(X)
            Pf = self.field_model.predict_proba(X)
            return Pd, Pf, Pf >= 0.5
        Pd = self.domain_cal.predict_proba(X)
        Pf = self.field_cal.transform(X)
        if variant == "calibrated":  # calibrated + tuned thresholds, no mask
            return Pd, Pf, Pf >= self.field_cal.thr
        pred_dom = classes[Pd.argmax(1)]
        Pf = Pf * domain_mask(pred_dom, self.vocab)
        pred = Pf >= self.field_cal.thr
        if variant == "full_min1":  # a search always has >= 1 field: fall back to the in-domain argmax
            empty = (~pred.any(1)) & (pred_dom != NONE_DOMAIN)
            pred[empty, Pf[empty].argmax(1)] = True
        return Pd, Pf, pred


VARIANTS = ("raw", "calibrated", "full", "full_min1")
