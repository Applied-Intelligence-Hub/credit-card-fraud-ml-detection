"""End-to-end experiment for one dataset (Sections 3 and 4 of the paper)."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .config import MODELS_PLAN, SHORT, Config
from .curves import pr_curve_df, pr_pgf, roc_curve_df, roc_pgf
from .data import prepare_data
from .ensembles import evaluate_ensembles, oof_predictions
from .explain import base_model_shap, ensemble_shap
from .metrics import evaluate_scores, operating_point_summary
from .models import fit_final
from .optimization import cv_threshold_stats, optimization_summary, run_optimization, trials_table

log = logging.getLogger("ccfd")


def _save(df: pd.DataFrame, path: str, **kw) -> None:
    df.to_csv(path, index=kw.pop("index", False), **kw)


def run_experiment(cfg: Config) -> Dict:
    t0 = time.time()
    out = cfg.output_dir
    for sub in ["", "models", "curves", "shap", "latex"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg.__dict__, fh, indent=2, default=str)

    # ---- Data -------------------------------------------------------------
    data = prepare_data(cfg)
    train, test = data["train"], data["test"]
    with open(os.path.join(out, "data_summary.json"), "w") as fh:
        json.dump({k: data[k] for k in ["quality_raw", "quality_modelled", "split"]}, fh, indent=2)
    _save(data["outliers_per_feature"], os.path.join(out, "outliers_per_feature.csv"), index=True)

    # ---- Optimization -----------------------------------------------------
    studies = run_optimization(train, cfg)
    trials = trials_table(studies)
    _save(trials, os.path.join(out, "optuna_trials.csv"))
    _save(optimization_summary(trials, cfg.n_startup_trials), os.path.join(out, "optimization_summary.csv"))
    best = {m: studies[m].best_params for m, _, _ in MODELS_PLAN}
    with open(os.path.join(out, "best_params.json"), "w") as fh:
        json.dump({m: {"params": best[m], "cv_ap": studies[m].best_value} for m in best}, fh, indent=2)

    # ---- CV thresholds + final refit + holdout scoring ---------------------
    rows, cv_stats, finals = [], {}, {}
    for m, sc, imb in MODELS_PLAN:
        st = cv_threshold_stats(m, best[m], sc, imb, train, cfg)
        cv_stats[m] = st
        fin = fit_final(m, best[m], sc, imb, train, test, cfg)
        finals[m] = fin
        joblib.dump(fin["model"], os.path.join(out, "models", f"{SHORT[m]}.pkl"))
        r = evaluate_scores(fin["y_test"], fin["scores"], st["th_mean"])
        rows.append({"Configuration": f"B: {SHORT[m]}", "CV_AP_mean": st["cv_ap_mean"],
                     "CV_AP_SD": st["cv_ap_std"], "Threshold_SD": st["th_std"], **r})
        log.info("%s | holdout AP %.4f | F2 %.4f | tau %.4f +/- %.4f",
                 m, r["AP"], r["F2"], st["th_mean"], st["th_std"])
    y_te = finals[MODELS_PLAN[0][0]]["y_test"]
    holdout = pd.DataFrame({m: finals[m]["scores"] for m, _, _ in MODELS_PLAN})
    _save(holdout.assign(Class=y_te), os.path.join(out, "holdout_scores.csv"))

    thr = pd.DataFrame([{"model": SHORT[m], "th_mean": s["th_mean"], "th_std": s["th_std"],
                         **{f"th_fold{i+1}": v for i, v in enumerate(s["th_folds"])}}
                        for m, s in cv_stats.items()])
    _save(thr, os.path.join(out, "threshold_stability.csv"))

    # ---- Ensembles --------------------------------------------------------
    oofp = oof_predictions(train, best, cfg)
    _save(oofp["oof"].assign(Class=oofp["labels"]), os.path.join(out, "oof_scores.csv"))
    cv_ap = {m: cv_stats[m]["cv_ap_mean"] for m, _, _ in MODELS_PLAN}
    base_th = {m: cv_stats[m]["th_mean"] for m, _, _ in MODELS_PLAN}
    ens = evaluate_ensembles(oofp, holdout, y_te, cv_ap, base_th, cfg)
    joblib.dump({"meta": ens["meta"], "scaler": ens["meta_scaler"]}, os.path.join(out, "models", "stacking.pkl"))

    results = pd.concat([pd.DataFrame(rows), ens["results"]], ignore_index=True)
    n_total = len(y_te)
    ops = pd.DataFrame([operating_point_summary(r, n_total) for r in results.to_dict("records")])
    results = pd.concat([results, ops], axis=1)
    results = results.sort_values("AP", ascending=False, kind="stable").reset_index(drop=True)
    results.insert(1, "Rank_AP", np.arange(1, len(results) + 1))
    _save(results, os.path.join(out, "holdout_results.csv"))

    ens_tab = pd.DataFrame({"weight": ens["weights"]})
    ens_tab.index = [SHORT[m] for m in ens_tab.index]

    # Kendall tau between AP and ROC-AUC rankings of the base learners
    base = results[results["Configuration"].str.startswith("B:")]
    tau = kendalltau(base["AP"], base["ROC-AUC"])
    gap = pd.DataFrame([{"model": SHORT[m], "cv_ap": cv_ap[m],
                         "holdout_ap": float(results.loc[results["Configuration"] == f"B: {SHORT[m]}", "AP"].iloc[0]),
                         "cv_ap_sd": cv_stats[m]["cv_ap_std"]} for m, _, _ in MODELS_PLAN])
    gap["gap"] = gap["holdout_ap"] - gap["cv_ap"]
    _save(gap, os.path.join(out, "cv_vs_holdout.csv"))

    # ---- Curves (exact) ---------------------------------------------------
    pgf = {}
    for m, _, _ in MODELS_PLAN:
        s = finals[m]["scores"]
        _save(pr_curve_df(y_te, s), os.path.join(out, "curves", f"pr_{SHORT[m]}.csv"))
        _save(roc_curve_df(y_te, s), os.path.join(out, "curves", f"roc_{SHORT[m]}.csv"))
        pgf[f"pr_{SHORT[m]}"] = pr_pgf(y_te, s)
        pgf[f"roc_{SHORT[m]}"] = roc_pgf(y_te, s)
    for m, g in trials.groupby("model", sort=False):
        pts = g[["trial", "best_so_far"]].to_numpy()
        keep = [0] + [i for i in range(1, len(pts)) if pts[i, 1] != pts[i - 1, 1]] + [len(pts) - 1]
        pgf[f"conv_{SHORT[m]}"] = " ".join(f"({int(pts[i,0])},{pts[i,1]:.4f})" for i in sorted(set(keep)))
    with open(os.path.join(out, "latex", "pgfplots_coordinates.json"), "w") as fh:
        json.dump(pgf, fh, indent=1)

    # ---- SHAP ---------------------------------------------------------------
    if cfg.run_shap:
        for m, _, _ in MODELS_PLAN:
            log.info("SHAP: %s", m)
            imp, _ = base_model_shap(m, finals[m]["model"], finals[m]["X_background"], finals[m]["X_eval"], cfg)
            _save(imp.head(cfg.shap_top_k) if cfg.shap_top_k else imp,
                  os.path.join(out, "shap", f"base_{SHORT[m]}.csv"))
        log.info("SHAP: ensembles")
        es = ensemble_shap(oofp["oof"], oofp["labels"], holdout, ens["weights"], base_th, cfg)
        for key in ["mean_abs", "normalized"]:
            t = es[key].copy()
            t.index = [SHORT[m] for m in t.index]
            _save(t, os.path.join(out, "shap", f"ensemble_{key}.csv"), index=True)
        ens_tab["stacking_proxy_coef"] = es["stacking_proxy_coef"].values
    _save(ens_tab, os.path.join(out, "ensemble_internals.csv"), index=True)

    summary = {"dataset": cfg.dataset, "n_holdout": n_total, "n_holdout_fraud": int(y_te.sum()),
               "kendall_tau_ap_vs_roc_base": float(tau.statistic),
               "runtime_min": (time.time() - t0) / 60}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Done in %.1f min. Results in %s", summary["runtime_min"], out)
    return {"results": results, "summary": summary}
