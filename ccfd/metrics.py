"""Metrics, F2 threshold selection, confusion counts and Wilson intervals."""
from __future__ import annotations

from math import sqrt
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (average_precision_score, fbeta_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def f2_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximizing F2 over all distinct scores (Eq. 8 of the paper)."""
    p, r, thr = precision_recall_curve(y_true, scores)
    if len(thr) == 0:
        return 0.5
    p, r = p[:-1], r[:-1]
    f2 = 5.0 * p * r / (4.0 * p + r + 1e-12)
    return float(thr[int(np.nanargmax(f2))])


def threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: Optional[float]) -> Dict:
    """AP and ROC-AUC (threshold-free) plus metrics at a fixed threshold.

    If threshold is None, `scores` are taken as binary decisions (majority vote).
    """
    y_pred = scores.astype(int) if threshold is None else (scores >= threshold).astype(int)
    out = {
        "AP": average_precision_score(y_true, scores),
        "ROC-AUC": roc_auc_score(y_true, scores),
        "Threshold": np.nan if threshold is None else float(threshold),
    }
    out.update(threshold_metrics(y_true, y_pred))
    out.update(confusion_counts(y_true, y_pred))
    return out


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def counts_from_metrics(precision: float, recall: float, n_pos: int) -> Tuple[int, int, int]:
    """Recover (TP, FP, FN) from reported precision/recall (used for the paper's Table 8)."""
    tp = int(round(recall * n_pos))
    alerts = int(round(tp / precision)) if precision > 0 else 0
    return tp, alerts - tp, n_pos - tp


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def operating_point_summary(row: Dict, n_total: int) -> Dict:
    """Alerts per 10k, FPR and Wilson intervals for recall and precision."""
    tp, fp, fn, tn = row["TP"], row["FP"], row["FN"], row["TN"]
    alerts = tp + fp
    r_lo, r_hi = wilson_interval(tp, tp + fn)
    p_lo, p_hi = wilson_interval(tp, alerts) if alerts else (np.nan, np.nan)
    return {
        "Alerts": alerts,
        "Alerts_per_10k": 1e4 * alerts / n_total,
        "FPR_pct": 100.0 * fp / max(fp + tn, 1),
        "Recall_CI_low": r_lo, "Recall_CI_high": r_hi,
        "Precision_CI_low": p_lo, "Precision_CI_high": p_hi,
    }
