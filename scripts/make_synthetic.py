#!/usr/bin/env python3
"""Create small synthetic CSVs with the DS1 and DS2 schemas, for smoke tests only.

They have no relation to real card data and must not be used to report results.

  python scripts/make_synthetic.py --out data_synthetic --n 20000
"""
import argparse
import os

import numpy as np
import pandas as pd


def make(n, prevalence, with_time, balanced, seed):
    rng = np.random.default_rng(seed)
    n1 = n // 2 if balanced else max(30, int(n * prevalence))
    n0 = n - n1
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    X = rng.normal(size=(n, 28)).astype(np.float32)
    shift = np.zeros(28)
    shift[[3, 13, 0, 7, 10]] = [2.0, -2.5, -1.0, 1.0, 1.5]   # make a few components informative
    X[y == 1] += shift * rng.uniform(0.6, 1.4, size=(n1, 1))
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = np.round(rng.lognormal(3.5, 1.3, size=n), 2)
    if with_time:
        t = np.sort(rng.uniform(0, 172800, size=n))
        df.insert(0, "Time", t)
    df["Class"] = y
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if not with_time:
        df.insert(0, "id", np.arange(n))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synthetic")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    make(a.n, 0.004, True, False, 1).to_csv(os.path.join(a.out, "synthetic_ds1.csv"), index=False)
    make(a.n * 3, 0.5, False, True, 2).to_csv(os.path.join(a.out, "synthetic_ds2.csv"), index=False)
    print("written to", a.out)
