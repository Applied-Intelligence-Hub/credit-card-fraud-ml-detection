#!/usr/bin/env python3
"""Cross-dataset comparison and SHAP agreement, from two finished runs.

  python scripts/compare_datasets.py --ds1 outputs/ds1 --ds2 outputs/ds2 --out outputs/compare
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccfd.compare import compare_runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds1", required=True)
    ap.add_argument("--ds2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = compare_runs(a.ds1, a.ds2, a.out)
    print(res["table"].round(3).to_string(index=False))
    print(json.dumps(res["stats"], indent=2))


if __name__ == "__main__":
    main()
