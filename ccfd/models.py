"""Model factory, search spaces and the fold-level train/predict routine."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import Config
from .features import apply_feature_engineering, apply_smote_safe, scale_features, split_xy


def build_model(name: str, params: Dict, imb_name: str, cfg: Config):
    rs = cfg.random_state
    if name == "Logistic Regression":
        return LogisticRegression(solver="lbfgs", max_iter=5000, n_jobs=cfg.n_jobs,
                                  C=params.get("C", 1.0), class_weight=None, random_state=rs)
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", probability=True, cache_size=2000,
                   C=params.get("C", 1.0), gamma=params.get("gamma", "scale"), random_state=rs)
    if name == "Random Forest":
        cw = "balanced" if imb_name.lower() == "class_weight" else None
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200), max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", "sqrt"),
            class_weight=cw, n_jobs=cfg.n_jobs, random_state=rs)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 300), max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1), subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 3),
            gamma=0.1, reg_alpha=0.0, reg_lambda=params.get("reg_lambda", 1.0),
            objective="binary:logistic", tree_method="hist", n_jobs=cfg.n_jobs, random_state=rs)
    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 300), num_leaves=params.get("num_leaves", 31),
            max_depth=params.get("max_depth", -1), learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8), colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 0.0), objective="binary", is_unbalance=False,
            n_jobs=1, random_state=rs, verbose=-1)
    if name == "CatBoost":
        extra = {"use_best_model": False} if cfg.catboost_disable_best_model else {}
        return CatBoostClassifier(
            iterations=params.get("iterations", 500), depth=params.get("depth", 6),
            learning_rate=params.get("learning_rate", 0.1), l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            loss_function="Logloss", auto_class_weights="Balanced",
            random_seed=rs, verbose=False, allow_writing_files=False, **extra)
    raise ValueError(f"Unknown model {name}")


def suggest_params(trial, name: str) -> Dict:
    """Optuna search spaces (Table 4 of the paper)."""
    if name == "Logistic Regression":
        return dict(C=trial.suggest_float("C", 1e-3, 1e3, log=True))
    if name == "SVM (RBF)":
        return dict(C=trial.suggest_float("C", 1e-2, 1e2, log=True),
                    gamma=trial.suggest_float("gamma", 1e-4, 1e-2, log=True))
    if name == "Random Forest":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]))
    if name == "XGBoost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0, log=True))
    if name == "LightGBM":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", 6, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 3.0))
    if name == "CatBoost":
        return dict(
            iterations=trial.suggest_int("iterations", 300, 1000, step=100),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0))
    raise ValueError(name)


def model_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def train_predict_fold(name: str, params: Dict, scaler_name: str, imb_name: str,
                       df_tr_raw: pd.DataFrame, df_va_raw: pd.DataFrame,
                       cfg: Config) -> Tuple[np.ndarray, pd.Series]:
    """Fit on one training fold and score its validation fold.

    Feature engineering, scaling and SMOTE are fitted on the training fold
    only; the validation fold is never resampled. As in the notebooks, the
    boosting models receive the validation fold as an evaluation set (used for
    monitoring only; see Config.catboost_disable_best_model).
    """
    df_tr = apply_feature_engineering(df_tr_raw)
    df_va = apply_feature_engineering(df_va_raw)
    X_tr, y_tr = split_xy(df_tr)
    X_va, y_va = split_xy(df_va)
    X_tr, X_va, _ = scale_features(X_tr, X_va, scaler_name)
    if imb_name.lower() == "smote":
        X_tr, y_tr = apply_smote_safe(X_tr, y_tr, cfg.random_state)
    model = build_model(name, params, imb_name, cfg)
    try:
        if name == "XGBoost":
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        elif name == "LightGBM":
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision")
        elif name == "CatBoost":
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        else:
            model.fit(X_tr, y_tr)
    except TypeError:
        model.fit(X_tr, y_tr)
    return model_scores(model, X_va), y_va


def fit_final(name: str, params: Dict, scaler_name: str, imb_name: str,
              df_train: pd.DataFrame, df_test: pd.DataFrame, cfg: Config) -> Dict:
    """Refit on the whole training partition and score the holdout once."""
    df_tr = apply_feature_engineering(df_train)
    df_te = apply_feature_engineering(df_test)
    X_tr, y_tr = split_xy(df_tr)
    X_te, y_te = split_xy(df_te)
    X_tr_s, X_te_s, scaler = scale_features(X_tr, X_te, scaler_name)
    if imb_name.lower() == "smote":
        X_fit, y_fit = apply_smote_safe(X_tr_s, y_tr, cfg.random_state)
    else:
        X_fit, y_fit = X_tr_s, y_tr
    model = build_model(name, params, imb_name, cfg)
    model.fit(X_fit, y_fit)
    return {
        "model": model,
        "scaler": scaler,
        "scores": model_scores(model, X_te_s),
        "y_test": y_te.values,
        "X_background": X_tr_s,   # pre-SMOTE training representation (SHAP background)
        "X_eval": X_te_s,         # holdout representation (SHAP evaluation)
    }
