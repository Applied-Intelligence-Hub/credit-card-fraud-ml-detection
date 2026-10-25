"""Out-of-fold predictions and the four ensemble strategies."""
from __future__ import annotations

import logging
import math
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from .config import MODELS_PLAN, Config
from .metrics import evaluate_scores, f2_optimal_threshold
from .models import train_predict_fold
from .optimization import make_cv

log = logging.getLogger("ccfd")


def oof_predictions(df_train: pd.DataFrame, best_params: Dict[str, Dict], cfg: Config) -> Dict:
    """Out-of-fold scores of every tuned learner on the training partition."""
    skf = make_cv(df_train, cfg)
    names = [m for m, _, _ in MODELS_PLAN]
    oof = {m: np.zeros(len(df_train), dtype=np.float32) for m in names}
    labels = np.zeros(len(df_train), dtype=np.int8)
    fold_ap = {m: [] for m in names}
    folds = list(skf.split(df_train, df_train["Class"]))
    for k, (tr_idx, va_idx) in enumerate(folds, 1):
        log.info("OOF fold %d/%d", k, len(folds))
        df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
        df_va = df_train.iloc[va_idx].reset_index(drop=True)
        labels[va_idx] = df_va["Class"].values
        for m, sc, imb in MODELS_PLAN:
            s, _ = train_predict_fold(m, best_params[m], sc, imb, df_tr, df_va, cfg)
            oof[m][va_idx] = s
            if len(np.unique(df_va["Class"])) > 1:
                fold_ap[m].append(average_precision_score(df_va["Class"].values, s))
    return {"oof": pd.DataFrame(oof), "labels": labels, "fold_ap": fold_ap, "folds": folds}


def evaluate_ensembles(oof_pack: Dict, holdout: pd.DataFrame, y_holdout: np.ndarray,
                       base_cv_ap: Dict[str, float], base_thresholds: Dict[str, float],
                       cfg: Config) -> Dict:
    """Mean, AP-weighted mean, majority vote and stacking, scored once on the holdout.

    All weights, thresholds and the meta-learner are fitted on training (OOF) data.
    """
    names = list(holdout.columns)
    oof, labels, folds = oof_pack["oof"], oof_pack["labels"], oof_pack["folds"]

    # Stacking meta-learner on standardized OOF scores
    scaler = StandardScaler().fit(oof)
    meta = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=cfg.random_state)
    meta.fit(scaler.transform(oof), labels)

    w = np.array([base_cv_ap[m] for m in names])
    w = w / (w.sum() + 1e-12)
    th_arr = np.array([base_thresholds[m] for m in names])
    need = math.ceil(len(names) / 2)  # 3 of 6; a 3-3 tie goes to fraud

    def mv(mat: np.ndarray) -> np.ndarray:
        return ((mat >= th_arr).sum(axis=1) >= need).astype(int)

    hold = {
        "Mean": holdout.mean(axis=1).values,
        "Weighted": holdout.values.dot(w),
        "Stacking": meta.predict_proba(scaler.transform(holdout))[:, 1],
        "Majority vote": mv(holdout.values),
    }
    oof_scores = {
        "Mean": oof.mean(axis=1).values,
        "Weighted": oof.values.dot(w),
        "Stacking": meta.predict_proba(scaler.transform(oof))[:, 1],
    }
    thresholds = {k: f2_optimal_threshold(labels, v) for k, v in oof_scores.items()}

    # Fold-level AP dispersion (development stage). For stacking this is computed
    # on the matrix the meta-learner was fitted on, so it may be optimistic.
    fold_ap = {k: [] for k in ["Mean", "Weighted", "Stacking", "Majority vote"]}
    for _, va_idx in folds:
        y = labels[va_idx]
        if len(np.unique(y)) < 2:
            continue
        sub = oof.iloc[va_idx]
        fold_ap["Mean"].append(average_precision_score(y, sub.mean(axis=1).values))
        fold_ap["Weighted"].append(average_precision_score(y, sub.values.dot(w)))
        fold_ap["Stacking"].append(average_precision_score(y, meta.predict_proba(scaler.transform(sub))[:, 1]))
        fold_ap["Majority vote"].append(average_precision_score(y, mv(sub.values)))

    rows = []
    for k in ["Mean", "Weighted", "Stacking"]:
        r = evaluate_scores(y_holdout, hold[k], thresholds[k])
        rows.append({"Configuration": f"E: {k}", "CV_AP_SD": float(np.std(fold_ap[k])), **r})
    r = evaluate_scores(y_holdout, hold["Majority vote"], None)  # AP/ROC from binary votes
    rows.append({"Configuration": "E: Majority vote", "CV_AP_SD": float(np.std(fold_ap["Majority vote"])), **r})

    return {
        "results": pd.DataFrame(rows),
        "weights": pd.Series(w, index=names, name="weight"),
        "thresholds": thresholds,
        "meta": meta, "meta_scaler": scaler,
        "holdout_scores": hold,
        "majority_need": need,
        "base_threshold_array": th_arr,
    }
