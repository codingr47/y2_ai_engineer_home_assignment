"""Metrics for the domain + field heads."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support

CONF_CUTS = (0.5, 0.7, 0.8, 0.9, 0.95)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(conf[m].mean() - correct[m].mean())
            total += m.mean() * gap
            table.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()),
                          "conf": round(float(conf[m].mean()), 4), "acc": round(float(correct[m].mean()), 4)})
    return float(total), table


def domain_metrics(y_true, Pd, classes) -> dict:
    y_true = np.asarray(y_true)
    y_pred = classes[Pd.argmax(1)]
    conf = Pd.max(1)
    correct = (y_pred == y_true).astype(float)
    onehot = (classes[None, :] == y_true[:, None]).astype(float)
    e, table = ece(conf, correct)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    return {
        "accuracy": float(correct.mean()),
        "ece": e,
        "brier": float(((Pd - onehot) ** 2).sum(1).mean()),
        "log_loss": float(log_loss(y_true, np.clip(Pd, 1e-7, 1), labels=classes)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion": {"labels": list(classes), "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist()},
        "reliability": table,
        "routing": [{"min_conf": t, "coverage": float((conf >= t).mean()),
                     "accuracy": float(correct[conf >= t].mean()) if (conf >= t).any() else None}
                    for t in CONF_CUTS],
    }


def field_metrics(Y, pred, vocab) -> dict:
    Y = np.asarray(Y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    support = Y.sum(0)
    present = np.flatnonzero(support > 0)
    p, r, f, _ = precision_recall_fscore_support(Y, pred, zero_division=0)
    per_field = sorted(
        ({"field": vocab[j], "recall": float(r[j]), "precision": float(p[j]), "f1": float(f[j]),
          "support": int(support[j])} for j in present),
        key=lambda d: (d["recall"], -d["support"]))
    return {
        "micro_f1": float(f1_score(Y, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y[:, present], pred[:, present], average="macro", zero_division=0)),
        "macro_over_n_fields": int(len(present)),
        "exact_set_accuracy": float((Y == pred).all(1).mean()),
        "per_field": per_field,
    }


def field_calibration(Y, Pf) -> dict:
    """Reliability of individual field probabilities (flattened over rows x fields)."""
    e, _ = ece(np.asarray(Pf).ravel(), np.asarray(Y).ravel().astype(float))
    return {"field_ece": e, "field_brier": float(((np.asarray(Pf) - Y) ** 2).mean())}


def breakdown(meta, key, y_dom, Pd, classes, Y, pred) -> dict:
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[m.get(key, "?")].append(i)
    out = {}
    for g, idx in sorted(groups.items()):
        idx = np.array(idx)
        out[g] = {
            "n": int(len(idx)),
            "domain_accuracy": float(accuracy_score(np.asarray(y_dom)[idx], classes[Pd[idx].argmax(1)])),
            "micro_f1": float(f1_score(Y[idx], pred[idx], average="micro", zero_division=0)),
            "exact_set_accuracy": float((Y[idx] == pred[idx]).all(1).mean()),
        }
    return out


def evaluate(heads, split, variant="full", detailed=True) -> dict:
    Pd, Pf, pred = heads.predict(split.X, variant)
    classes = heads.domain_model.classes_
    Y = heads.mlb.transform(split.fields)
    dm = domain_metrics(split.domains, Pd, classes)
    fm = field_metrics(Y, pred, heads.vocab)
    res = {
        "variant": variant, "split": split.name, "n": len(split.labels),
        "domain_accuracy": dm["accuracy"], "domain_ece": dm["ece"], "domain_brier": dm["brier"],
        "domain_log_loss": dm["log_loss"],
        "micro_f1": fm["micro_f1"], "macro_f1": fm["macro_f1"], "exact_set_accuracy": fm["exact_set_accuracy"],
        **field_calibration(Y, Pf),
    }
    if detailed:
        res["domain"] = dm
        res["per_field"] = fm["per_field"]
        res["macro_over_n_fields"] = fm["macro_over_n_fields"]
        res["by_language"] = breakdown(split.meta, "language", split.domains, Pd, classes, Y, pred)
        res["by_complexity"] = breakdown(split.meta, "complexity", split.domains, Pd, classes, Y, pred)
        res["by_quality"] = breakdown(split.meta, "quality", split.domains, Pd, classes, Y, pred)
    return res


HEADLINE = ("domain_accuracy", "micro_f1", "macro_f1", "exact_set_accuracy", "domain_ece", "field_ece")
