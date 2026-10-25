import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccfd.config import default_config  # noqa: E402
from ccfd.data import match_prevalence  # noqa: E402
from ccfd.features import apply_feature_engineering, scale_features  # noqa: E402
from ccfd.metrics import (counts_from_metrics, evaluate_scores,  # noqa: E402
                          f2_optimal_threshold, wilson_interval)


def test_counts_match_paper_table8():
    assert counts_from_metrics(0.903226, 0.857143, 98) == (84, 9, 14)   # DS1 LightGBM
    assert counts_from_metrics(0.704348, 0.826531, 98) == (81, 34, 17)  # DS1 LR
    assert counts_from_metrics(0.870000, 0.887755, 98) == (87, 13, 11)  # DS2 XGBoost


def test_wilson_interval():
    lo, hi = wilson_interval(84, 98)
    assert round(lo, 3) == 0.774 and round(hi, 3) == 0.913
    lo, hi = wilson_interval(81, 115)
    assert round(lo, 2) == 0.62 and round(hi, 2) == 0.78


def test_prevalence_matching_reproduces_ds2_counts():
    df = pd.DataFrame({"x": np.zeros(284315 * 2, dtype=np.float32),
                       "Class": np.r_[np.zeros(284315, int), np.ones(284315, int)]})
    out = match_prevalence(df, 0.00172, 42)
    assert int(out["Class"].sum()) == 489 and len(out) == 284804


def test_cyclical_features():
    df = pd.DataFrame({"Time": [0.0, 6 * 3600.0, 30 * 3600.0], "Amount": [0.0, 10.0, 1.0], "Class": [0, 1, 0]})
    fe = apply_feature_engineering(df)
    assert set(fe.columns) == {"Amount_log", "Hour_sin", "Hour_cos", "Class"}
    assert np.isclose(fe.loc[1, "Hour_sin"], 1.0, atol=1e-6)
    assert np.isclose(fe.loc[2, "Hour_sin"], 1.0, atol=1e-6)   # 30 h wraps to 6 h
    assert fe.loc[0, "Amount_log"] == 0.0


def test_scaler_is_fitted_on_training_only():
    tr = pd.DataFrame({"Amount_log": [0.0, 1.0], "V1": [5.0, 6.0]})
    te = pd.DataFrame({"Amount_log": [2.0], "V1": [7.0]})
    tr_s, te_s, _ = scale_features(tr, te, "MinMaxScaler")
    assert te_s.loc[0, "Amount_log"] == 2.0          # outside [0, 1]: not refitted on test
    assert te_s.loc[0, "V1"] == 7.0                  # anonymized components are not scaled


def test_f2_threshold_and_vote_metrics():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    assert f2_optimal_threshold(y, s) == 0.7
    r = evaluate_scores(y, np.array([0, 0, 1, 1, 1]), None)
    assert (r["TP"], r["FP"], r["FN"]) == (2, 1, 0)


def test_default_configs():
    c1 = default_config("ds1", "a.csv", "o")
    c2 = default_config("ds2", "b.csv", "o")
    assert c1.target_fraud_rate is None and c2.target_fraud_rate == 0.00172
    assert c1.n_trials["SVM (RBF)"] == 40 and c2.n_trials["Logistic Regression"] == 80


def test_wilson_rounding_83_of_98():
    lo, hi = wilson_interval(83, 98)
    assert f"{lo:.2f}" == "0.76" and f"{hi:.2f}" == "0.90"   # 0.9050 -> .90 (Table 8)
