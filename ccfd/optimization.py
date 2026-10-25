"""Optuna/TPE tuning on mean cross-validated AP, and CV threshold statistics."""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .config import MODELS_PLAN, Config
from .metrics import f2_optimal_threshold
from .models import suggest_params, train_predict_fold

log = logging.getLogger("ccfd")


def make_cv(df_train: pd.DataFrame, cfg: Config) -> StratifiedKFold:
    """J = max(2, min(5, N+)) stratified folds (Eq. 7)."""
    n_pos = int((df_train["Class"] == 1).sum())
    n_splits = max(2, min(cfg.n_splits_max, n_pos))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)


def run_optimization(df_train: pd.DataFrame, cfg: Config) -> Dict[str, optuna.Study]:
    """One study per learner, sharing a single seeded TPE sampler (notebook behaviour)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=cfg.random_state, n_startup_trials=cfg.n_startup_trials)
    skf = make_cv(df_train, cfg)
    folds = list(skf.split(df_train, df_train["Class"]))
    studies = {}
    for name, scaler_name, imb_name in MODELS_PLAN:
        n_trials = cfg.n_trials.get(name, 50)
        log.info("[%s] Optuna: %d trials | scaler=%s | imbalance=%s", name, n_trials, scaler_name, imb_name)

        def objective(trial, name=name, scaler_name=scaler_name, imb_name=imb_name):
            params = suggest_params(trial, name)
            aps = []
            for tr_idx, va_idx in folds:
                s, y = train_predict_fold(name, params, scaler_name, imb_name,
                                          df_train.iloc[tr_idx].copy(), df_train.iloc[va_idx].copy(), cfg)
                aps.append(average_precision_score(y, s))
            return float(np.mean(aps))

        study = optuna.create_study(direction="maximize", study_name=f"{name} Study", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        log.info("[%s] best CV AP %.6f | %s", name, study.best_value, study.best_params)
        studies[name] = study
    return studies


def trials_table(studies: Dict[str, optuna.Study]) -> pd.DataFrame:
    rows = []
    for name, st in studies.items():
        best_so_far = -np.inf
        for t in st.trials:
            if t.value is None:
                continue
            best_so_far = max(best_so_far, t.value)
            rows.append({"model": name, "trial": t.number + 1, "value": t.value,
                         "best_so_far": best_so_far,
                         "duration_s": t.duration.total_seconds() if t.duration else np.nan,
                         **{f"param_{k}": v for k, v in t.params.items()}})
    return pd.DataFrame(rows)


def optimization_summary(trials: pd.DataFrame, n_startup: int) -> pd.DataFrame:
    """Best value, trial of the best, median, range, and TPE gain over the random phase (Table 5)."""
    out = []
    for name, g in trials.groupby("model", sort=False):
        v = g["value"].to_numpy()
        out.append({
            "model": name, "n_trials": len(v), "best": v.max(),
            "best_trial": int(g["trial"].iloc[int(np.argmax(v))]),
            "median": float(np.median(v)), "min": v.min(), "max": v.max(),
            "best_random_phase": v[:n_startup].max(),
            "tpe_gain": v.max() - v[:n_startup].max(),
        })
    return pd.DataFrame(out)


def cv_threshold_stats(name: str, params: Dict, scaler_name: str, imb_name: str,
                       df_train: pd.DataFrame, cfg: Config) -> Dict:
    """Per-fold F2-optimal thresholds and AP; their mean and (population) SD."""
    skf = make_cv(df_train, cfg)
    ths, aps = [], []
    for tr_idx, va_idx in skf.split(df_train, df_train["Class"]):
        s, y = train_predict_fold(name, params, scaler_name, imb_name,
                                  df_train.iloc[tr_idx].copy(), df_train.iloc[va_idx].copy(), cfg)
        ths.append(f2_optimal_threshold(y.values, s))
        aps.append(average_precision_score(y, s))
    return {"th_mean": float(np.mean(ths)), "th_std": float(np.std(ths)), "th_folds": ths,
            "cv_ap_mean": float(np.mean(aps)), "cv_ap_std": float(np.std(aps)), "cv_ap_folds": aps}
