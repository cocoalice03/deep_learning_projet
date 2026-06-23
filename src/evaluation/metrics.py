import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y_true, y_prob, average):
    try:
        return float(roc_auc_score(y_true, y_prob, average=average))
    except Exception:
        return float("nan")


def _safe_average_precision(y_true, y_prob, average):
    try:
        return float(average_precision_score(y_true, y_prob, average=average))
    except Exception:
        return float("nan")


def multilabel_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "auc_macro": _safe_auc(y_true, y_prob, "macro"),
        "auc_micro": _safe_auc(y_true, y_prob, "micro"),
        "average_precision_macro": _safe_average_precision(y_true, y_prob, "macro"),
        "average_precision_micro": _safe_average_precision(y_true, y_prob, "micro"),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }
    return out


def per_label_metrics(y_true, y_prob, labels, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    for i, label in enumerate(labels):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        yd = y_pred[:, i]
        rows.append(
            {
                "label": label,
                "support": int(yt.sum()),
                "prevalence": float(yt.mean()),
                "auc": _safe_auc(yt, yp, None),
                "average_precision": _safe_average_precision(yt, yp, None),
                "f1": float(f1_score(yt, yd, zero_division=0)),
                "precision": float(precision_score(yt, yd, zero_division=0)),
                "recall": float(recall_score(yt, yd, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def find_best_threshold(y_true, y_prob, metric="f1_macro", thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    best_threshold = 0.5
    best_score = -1.0
    rows = []
    for threshold in thresholds:
        metrics = multilabel_metrics(y_true, y_prob, threshold=float(threshold))
        score = metrics.get(metric, float("nan"))
        rows.append({"threshold": float(threshold), metric: score})
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, pd.DataFrame(rows)
