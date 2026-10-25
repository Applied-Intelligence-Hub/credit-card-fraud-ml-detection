"""Data loading, quality checks, prevalence matching and the train/holdout split."""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .config import Config

log = logging.getLogger("ccfd")


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Read the CSV, cast types and drop the DS2 row identifier."""
    df = pd.read_csv(cfg.csv_path)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column with 0/1 labels.")
    with open(cfg.csv_path, "rb") as fh:  # guard against Git LFS pointer files
        head = fh.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise ValueError(f"{cfg.csv_path} is a Git LFS pointer; run 'git lfs pull' first.")
    df["Class"] = df["Class"].astype(np.int8)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        log.info("Dropped 'id' column (row identifier, not a feature)")
    for col in [c for c in df.columns if c != "Class"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    log.info("Dataset shape: %s | frauds=%d", df.shape, int(df["Class"].sum()))
    return df


def quality_report(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """Missing values, duplicates and 1.5-IQR outliers (computed on raw predictors)."""
    feats = [c for c in df.columns if c != "Class"]
    q1, q3 = df[feats].quantile(0.25), df[feats].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[feats] < lo) | (df[feats] > hi)
    n = len(df)
    n0 = int((df["Class"] == 0).sum())
    n1 = int((df["Class"] == 1).sum())
    summary = {
        "n_rows": n,
        "n_legit": n0,
        "n_fraud": n1,
        "prevalence_pct": 100.0 * n1 / n,
        "imbalance_ratio": n0 / max(n1, 1),
        "n_predictors": len(feats),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "rows_with_iqr_outlier": int(mask.any(axis=1).sum()),
        "rows_with_iqr_outlier_pct": 100.0 * float(mask.any(axis=1).mean()),
    }
    if "Amount" in df.columns:
        summary["amount_mean"] = float(df["Amount"].mean())
        summary["amount_max"] = float(df["Amount"].max())
    per_feature = pd.DataFrame({
        "lower_bound": lo,
        "upper_bound": hi,
        "n_outliers": mask.sum(),
        "pct_outliers": (100.0 * mask.sum() / n).round(2),
    }).sort_values("n_outliers", ascending=False)
    return summary, per_feature


def match_prevalence(df: pd.DataFrame, target_rate: float, random_state: int) -> pd.DataFrame:
    """Keep every legitimate row and undersample frauds to the target prevalence.

    Mirrors the DS2 notebook: n_fraud = int(n_legit * r / (1 - r)), frauds sampled
    with the fixed seed, then the concatenation is shuffled with the same seed.
    """
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]
    n_target = int(len(legit) * target_rate / (1.0 - target_rate))
    n_target = min(n_target, len(fraud))
    keep = fraud.sample(n=n_target, random_state=random_state)
    out = (pd.concat([legit, keep])
             .sample(frac=1, random_state=random_state)
             .reset_index(drop=True))
    log.info("Prevalence matched: target %.3f%% -> actual %.4f%% (%d frauds, %d rows)",
             100 * target_rate, 100 * out["Class"].mean(), n_target, len(out))
    return out


def stratified_split(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    tr_idx, te_idx = next(sss.split(df, df["Class"]))
    train = df.iloc[tr_idx].reset_index(drop=True)
    test = df.iloc[te_idx].reset_index(drop=True)
    log.info("Train=%d (frauds=%d) | Holdout=%d (frauds=%d)",
             len(train), int(train["Class"].sum()), len(test), int(test["Class"].sum()))
    return train, test


def prepare_data(cfg: Config) -> Dict:
    """Full data stage: load -> quality report -> (prevalence match) -> split."""
    raw = load_dataset(cfg)
    q_raw, q_feat = quality_report(raw)
    df = raw
    if cfg.target_fraud_rate is not None:
        df = match_prevalence(raw, cfg.target_fraud_rate, cfg.random_state)
    if cfg.legit_subsample is not None:
        legit = df[df["Class"] == 0].sample(frac=cfg.legit_subsample, random_state=cfg.random_state)
        df = pd.concat([legit, df[df["Class"] == 1]]).sample(frac=1, random_state=cfg.random_state)
        df = df.reset_index(drop=True)
        log.warning("Smoke-test subsample: %d rows kept (not for reporting)", len(df))
    if cfg.dedup_before_split:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        log.info("Dropped %d duplicate rows before splitting", before - len(df))
    q_model, _ = quality_report(df)
    train, test = stratified_split(df, cfg)
    return {
        "train": train,
        "test": test,
        "quality_raw": q_raw,
        "quality_modelled": q_model,
        "outliers_per_feature": q_feat,
        "split": {
            "n_train": len(train), "n_train_fraud": int(train["Class"].sum()),
            "n_test": len(test), "n_test_fraud": int(test["Class"].sum()),
        },
    }
