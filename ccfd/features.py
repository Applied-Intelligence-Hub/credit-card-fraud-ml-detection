"""Feature engineering, conditional scaling and fold-local SMOTE."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALERS = {
    "None (no scaling)": None,
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
}
# Only the monetary feature is scaled; anonymized components and cyclical
# terms are kept on their supplied scales.
COLS_TO_SCALE = ["Amount_log"]


def apply_feature_engineering(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Amount -> log(1 + Amount); Time (DS1 only) -> cyclical hour (sin, cos).

    Row-wise and stateless, so it cannot leak information across partitions.
    """
    df = raw_df.copy()
    if "Amount" in df.columns:
        df["Amount_log"] = np.log1p(df["Amount"].astype(np.float32))
    if "Time" in df.columns:
        hours = df["Time"].astype(np.float32) / 3600.0
        hour_of_day = (hours % 24.0).astype(np.float32)
        w = (2.0 * np.pi) / 24.0
        df["Hour_sin"] = np.sin(w * hour_of_day).astype(np.float32)
        df["Hour_cos"] = np.cos(w * hour_of_day).astype(np.float32)
    df = df.drop(columns=["Time", "Amount"], errors="ignore")
    for col in df.columns:
        if col == "Class":
            df[col] = df[col].astype(np.int8)
        elif df[col].dtype.kind in "fc":
            df[col] = df[col].astype(np.float32)
    return df


def scale_features(X_train: pd.DataFrame, X_other: pd.DataFrame,
                   scaler_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[object]]:
    """Fit the scaler on the training part only and apply it to the other part."""
    X_tr = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_ot = X_other.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler_cls = SCALERS[scaler_name]
    cols = [c for c in COLS_TO_SCALE if c in X_tr.columns]
    if scaler_cls is None or not cols:
        return X_tr, X_ot, None
    scaler = scaler_cls().fit(X_tr[cols])
    X_tr = X_tr.copy()
    X_ot = X_ot.copy()
    X_tr[cols] = scaler.transform(X_tr[cols])
    X_ot[cols] = scaler.transform(X_ot[cols])
    return X_tr, X_ot, scaler


def apply_smote_safe(X: pd.DataFrame, y: pd.Series, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    """SMOTE with k = max(1, min(5, N+ - 1)); only ever called on training folds."""
    n_pos = int((y == 1).sum())
    if n_pos < 2:
        return X, y
    k = max(1, min(5, n_pos - 1))
    X_res, y_res = SMOTE(random_state=random_state, k_neighbors=k,
                         sampling_strategy="auto").fit_resample(X, y)
    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X.columns)
    return X_res, y_res


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["Class"]), df["Class"].astype(int)
