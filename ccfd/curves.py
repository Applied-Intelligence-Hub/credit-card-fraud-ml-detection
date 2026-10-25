"""Exact PR/ROC curve export and pgfplots coordinate strings for the LaTeX figures."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def pr_curve_df(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    p, r, t = precision_recall_curve(y, s)
    return pd.DataFrame({"recall": r, "precision": p, "threshold": np.append(t, np.nan)})


def roc_curve_df(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    fpr, tpr, t = roc_curve(y, s)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": t})


def rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification (iterative, keeps endpoints)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3 or eps <= 0:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        d = b - a
        n = np.hypot(*d)
        seg = pts[i + 1:j]
        dist = (np.hypot(*(seg - a).T) if n == 0
                else np.abs(d[0] * (seg[:, 1] - a[1]) - d[1] * (seg[:, 0] - a[0])) / n)
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return pts[keep]


def pgf_coords(xy: Iterable[Tuple[float, float]], digits: int = 4) -> str:
    return " ".join(f"({x:.{digits}f},{y:.{digits}f})" for x, y in xy)


def pr_pgf(y: np.ndarray, s: np.ndarray, eps: float = 0.001) -> str:
    """Coordinates for `\\addplot coordinates {...}` (x=recall, y=precision).

    The curve is drawn as a step function in recall order; RDP only removes
    points that are collinear within `eps`.
    """
    df = pr_curve_df(y, s).iloc[::-1]  # increasing recall
    return pgf_coords(rdp(df[["recall", "precision"]].to_numpy(), eps))


def roc_pgf(y: np.ndarray, s: np.ndarray, eps: float = 0.0005) -> str:
    df = roc_curve_df(y, s)
    return pgf_coords(rdp(df[["fpr", "tpr"]].to_numpy(), eps))
