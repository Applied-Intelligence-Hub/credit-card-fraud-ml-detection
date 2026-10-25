#!/usr/bin/env python3
"""Rebuild Table 8 of the paper (confusion counts, alerts per 10k, Wilson intervals)
from the precision and recall reported in Tables 6 and 7. No data needed.

  python scripts/paper_operating_points.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccfd.metrics import counts_from_metrics, wilson_interval  # noqa: E402

# (precision, recall) at the selected thresholds; full precision where the notebooks printed it
DS1 = {"LightGBM": (0.903226, 0.857143), "XGBoost": (0.891304, 0.836735), "CatBoost": (0.807692, 0.857143),
       "RF": (0.803738, 0.877551), "SVM": (0.868132, 0.806122), "LR": (0.704348, 0.826531),
       "E: Mean": (0.856, 0.847), "E: Weighted": (0.856, 0.847), "E: Stacking": (0.856, 0.847),
       "E: Majority vote": (0.824, 0.857)}
DS2 = {"XGBoost": (0.870000, 0.887755), "LightGBM": (0.876289, 0.867347), "SVM": (0.876289, 0.867347),
       "CatBoost": (0.845361, 0.836735), "RF": (0.838384, 0.846939), "LR": (0.795918, 0.795918),
       "E: Mean": (0.842, 0.867), "E: Weighted": (0.843, 0.878), "E: Stacking": (0.867, 0.867),
       "E: Majority vote": (0.847, 0.847)}
N_TOTAL = {"DS1": 56962, "DS2": 56961}
N_POS = 98


def table(name, d):
    rows = []
    for m, (p, r) in d.items():
        tp, fp, fn = counts_from_metrics(p, r, N_POS)
        rl, rh = wilson_interval(tp, N_POS)
        pl, ph = wilson_interval(tp, tp + fp)
        rows.append({"Dataset": name, "Configuration": m, "TP": tp, "FN": fn, "FP": fp,
                     "Alerts/10k": round(1e4 * (tp + fp) / N_TOTAL[name], 1),
                     "Recall CI": f"{rl:.2f}-{rh:.2f}", "Precision CI": f"{pl:.2f}-{ph:.2f}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(pd.concat([table("DS1", DS1), table("DS2", DS2)]).to_string(index=False))
