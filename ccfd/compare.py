"""Cross-dataset analysis (Section 4.9) and explanation agreement (Section 4.8)."""
from __future__ import annotations

import itertools
import os
from collections import Counter
from typing import Dict, List

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .config import MODELS_PLAN, SHORT

BASE = [SHORT[m] for m, _, _ in MODELS_PLAN]


def load_shap_tables(run_dir: str) -> Dict[str, pd.DataFrame]:
    return {m: pd.read_csv(os.path.join(run_dir, "shap", f"base_{m}.csv")) for m in BASE
            if os.path.exists(os.path.join(run_dir, "shap", f"base_{m}.csv"))}


def topk_frequency(tables: Dict[str, pd.DataFrame], k: int) -> pd.Series:
    """In how many learners does each feature reach the top k?"""
    c = Counter(f for t in tables.values() for f in t["Feature"].head(k))
    return pd.Series(c, name=f"top{k}_count").sort_values(ascending=False)


def jaccard_matrix(tables: Dict[str, pd.DataFrame], k: int) -> pd.DataFrame:
    names = list(tables)
    mat = pd.DataFrame(1.0, index=names, columns=names)
    for a, b in itertools.combinations(names, 2):
        A, B = set(tables[a]["Feature"].head(k)), set(tables[b]["Feature"].head(k))
        mat.loc[a, b] = mat.loc[b, a] = len(A & B) / len(A | B)
    return mat


def rank_matrix(tables: Dict[str, pd.DataFrame], k_rows: int = 5) -> pd.DataFrame:
    """Rank of every feature that is top-`k_rows` for at least one learner (Fig. 12)."""
    rows = topk_frequency(tables, k_rows).index
    data = {m: {f: (int(t.index[t["Feature"] == f][0]) + 1 if f in set(t["Feature"]) else None)
                for f in rows} for m, t in tables.items()}
    return pd.DataFrame(data).loc[rows]


def compare_runs(ds1_dir: str, ds2_dir: str, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    r1 = pd.read_csv(os.path.join(ds1_dir, "holdout_results.csv"))
    r2 = pd.read_csv(os.path.join(ds2_dir, "holdout_results.csv"))
    m = r1[["Configuration", "AP", "Rank_AP"]].merge(
        r2[["Configuration", "AP", "Rank_AP"]], on="Configuration", suffixes=("_DS1", "_DS2"))
    m["Delta_AP"] = m["AP_DS2"] - m["AP_DS1"]
    m = m.sort_values("AP_DS2", ascending=False)
    m.to_csv(os.path.join(out_dir, "cross_dataset_ap.csv"), index=False)
    # Rounded to three decimals, as reported in the paper (ties allowed)
    a1, a2 = m["AP_DS1"].round(3), m["AP_DS2"].round(3)
    stats = {
        "kendall_tau_ranking": float(kendalltau(a1, a2).statistic),
        "spearman_rho_ranking": float(spearmanr(a1, a2).statistic),
    }
    agreement = {}
    for tag, d in [("ds1", ds1_dir), ("ds2", ds2_dir)]:
        tabs = load_shap_tables(d)
        if not tabs:
            continue
        for k in (5, 10):
            topk_frequency(tabs, k).to_csv(os.path.join(out_dir, f"{tag}_top{k}_frequency.csv"))
            J = jaccard_matrix(tabs, k)
            J.to_csv(os.path.join(out_dir, f"{tag}_jaccard_top{k}.csv"))
            pairs = [J.loc[a, b] for a, b in itertools.combinations(J.index, 2)]
            trio = [J.loc[a, b] for a, b in itertools.combinations(["XGBoost", "LightGBM", "CatBoost"], 2)
                    if a in J.index and b in J.index]
            agreement[f"{tag}_mean_jaccard_top{k}_all_pairs"] = sum(pairs) / len(pairs)
            if trio:
                agreement[f"{tag}_mean_jaccard_top{k}_boosting"] = sum(trio) / len(trio)
        rank_matrix(tabs).to_csv(os.path.join(out_dir, f"{tag}_rank_matrix.csv"))
    stats.update(agreement)
    pd.Series(stats).to_json(os.path.join(out_dir, "cross_dataset_stats.json"), indent=2)
    return {"table": m, "stats": stats}
