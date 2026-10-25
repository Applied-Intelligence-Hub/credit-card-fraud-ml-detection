#!/usr/bin/env python3
"""Quick-look PNG figures from a finished run (the paper itself uses TikZ).

  python scripts/quicklook.py --run outputs/ds1
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BASE = ["LightGBM", "XGBoost", "CatBoost", "RF", "SVM", "LR"]
COL = {"LightGBM": "#0072B2", "XGBoost": "#D55E00", "CatBoost": "#009E73",
       "RF": "#CC79A7", "SVM": "#E69F00", "LR": "#56B4E9"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    out = os.path.join(a.run, "figures")
    os.makedirs(out, exist_ok=True)
    res = pd.read_csv(os.path.join(a.run, "holdout_results.csv"))
    ap_of = {r["Configuration"][3:]: r["AP"] for _, r in res.iterrows()}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in BASE:
        pr = pd.read_csv(os.path.join(a.run, "curves", f"pr_{m}.csv"))
        roc = pd.read_csv(os.path.join(a.run, "curves", f"roc_{m}.csv"))
        axes[0].step(pr["recall"], pr["precision"], where="post", color=COL[m], label=f"{m} (AP={ap_of[m]:.3f})")
        axes[1].plot(roc["fpr"], roc["tpr"], color=COL[m], label=m)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (holdout)")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (holdout)", xscale="log")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "curves.png"), dpi=150)

    tr = pd.read_csv(os.path.join(a.run, "optuna_trials.csv"))
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, g in tr.groupby("model", sort=False):
        ax.step(g["trial"], g["best_so_far"], where="post", label=m)
    ax.set(xlabel="Trial", ylabel="Best mean CV AP so far", title="Optuna convergence")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "convergence.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    r = res.iloc[::-1]
    ax.barh(r["Configuration"], -r["FN"], color="#D55E00", label="missed frauds (FN)")
    ax.barh(r["Configuration"], r["FP"], color="#0072B2", label="false alarms (FP)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="count", title="Errors at the selected thresholds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "errors.png"), dpi=150)
    print("figures written to", out)


if __name__ == "__main__":
    main()
