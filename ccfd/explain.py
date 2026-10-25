"""SHAP explanations for base learners and explanatory proxies of the ensembles."""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer

from .config import Config

log = logging.getLogger("ccfd")
TREE_MODELS = {"XGBoost", "LightGBM", "CatBoost"}


def _as_2d(values) -> np.ndarray:
    arr = np.asarray(values)
    if isinstance(values, list):
        arr = np.asarray(values[1] if len(values) > 1 else values[0])
    if arr.ndim == 3:            # (n, features, classes) in recent SHAP versions
        arr = arr[..., -1]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def base_model_shap(name: str, model, X_bg: pd.DataFrame, X_eval: pd.DataFrame,
                    cfg: Config) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Global importance (mean and SD of |SHAP|) with the explainer used in the paper."""
    rs = cfg.random_state
    np.random.seed(rs)  # KernelExplainer draws coalitions from numpy's global RNG
    X = X_eval
    if name in TREE_MODELS and len(X) > cfg.tree_eval_max:
        X = X.sample(cfg.tree_eval_max, random_state=rs)
    if name in {"Random Forest", "SVM (RBF)"} and len(X) > cfg.svm_eval_max:
        X = X.sample(cfg.svm_eval_max, random_state=rs)
    if name == "SVM (RBF)" and not cfg.do_svm_shap:
        return pd.DataFrame(columns=["Feature", "MeanAbsSHAP", "SDAbsSHAP"]), None

    if name in TREE_MODELS:
        bg = X_bg if cfg.tree_bg_max is None or len(X_bg) <= cfg.tree_bg_max \
            else X_bg.sample(cfg.tree_bg_max, random_state=rs)
        expl = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        sv = expl.shap_values(X)
    elif name == "Random Forest":
        bg = shap.sample(X_bg, min(cfg.svm_bg_max, len(X_bg)), random_state=rs)

        def rf_logit(x, m=model):
            p = np.clip(m.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        sv = shap.KernelExplainer(rf_logit, bg).shap_values(X, nsamples=cfg.shap_nsamples, silent=True)
    elif name == "Logistic Regression":
        expl = shap.LinearExplainer(model, masker=shap.maskers.Independent(X_bg))
        sv = expl.shap_values(X)
    else:  # SVM: signed decision function
        bg = shap.sample(X_bg, min(cfg.svm_bg_max, len(X_bg)), random_state=rs)
        sv = shap.KernelExplainer(lambda x: model.decision_function(x), bg) \
            .shap_values(X, nsamples=cfg.shap_nsamples, silent=True)

    arr = _as_2d(sv)
    imp = (pd.DataFrame({"Feature": X.columns,
                         "MeanAbsSHAP": np.abs(arr).mean(axis=0),
                         "SDAbsSHAP": np.abs(arr).std(axis=0)})
           .sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True))
    imp.insert(0, "Rank", np.arange(1, len(imp) + 1))
    return imp, arr


def ensemble_shap(oof: pd.DataFrame, labels: np.ndarray, holdout: pd.DataFrame,
                  weights: pd.Series, base_thresholds: Dict[str, float], cfg: Config) -> Dict:
    """Explanatory proxies of the ensembles; each SHAP dimension is a base learner.

    Mean/Weighted: quantile-normalized scores, KernelExplainer vs 50 k-means centroids.
    Majority vote: raw scores (its thresholds live in raw space).
    Stacking: separate LR fitted on quantile-normalized OOF scores, LinearExplainer.
    """
    rs = cfg.random_state
    names = list(oof.columns)
    qt = QuantileTransformer(output_distribution="uniform",
                             n_quantiles=min(1000, len(oof)), random_state=rs).fit(oof)
    oof_q = pd.DataFrame(qt.transform(oof), columns=names)
    hold_q = pd.DataFrame(qt.transform(holdout), columns=names)
    np.random.seed(rs)
    bg_q = shap.kmeans(oof_q, cfg.ens_kmeans)
    bg_raw = shap.kmeans(oof, cfg.ens_kmeans)
    ev_q = hold_q.sample(min(cfg.ens_eval_max, len(hold_q)), random_state=rs)
    ev_raw = holdout.sample(min(cfg.ens_eval_max, len(holdout)), random_state=rs)

    w = weights.loc[names].values
    th = np.array([base_thresholds[m] for m in names])
    need = math.ceil(len(names) / 2)

    def kernel(fn, bg, ev):
        sv = shap.KernelExplainer(fn, bg).shap_values(ev, nsamples=cfg.ens_nsamples, silent=True)
        return np.abs(_as_2d(sv)).mean(axis=0)

    out = {}
    out["Mean"] = kernel(lambda x: x.mean(axis=1), bg_q, ev_q)
    out["Weighted"] = kernel(lambda x: x.dot(w), bg_q, ev_q)
    out["Majority vote"] = kernel(lambda x: ((x >= th).sum(axis=1) >= need).astype(int), bg_raw, ev_raw)

    proxy = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=rs).fit(oof_q.values, labels)
    lin = shap.LinearExplainer(proxy, masker=shap.maskers.Independent(oof_q.values))
    out["Stacking"] = np.abs(_as_2d(lin.shap_values(hold_q.values))).mean(axis=0)

    raw = pd.DataFrame(out, index=names)
    norm = raw / raw.sum(axis=0).replace(0, np.nan)
    coefs = pd.Series(proxy.coef_[0], index=names, name="stacking_proxy_coef")
    return {"mean_abs": raw, "normalized": norm, "stacking_proxy_coef": coefs}
