#!/usr/bin/env python3
"""Run the full pipeline for one dataset.

Examples
--------
  python scripts/run_experiment.py --dataset ds1 --data data/creditcard_2013.csv --out outputs/ds1
  python scripts/run_experiment.py --dataset ds2 --data data/creditcard_2023.csv --out outputs/ds2
  python scripts/run_experiment.py --dataset ds1 --data data/creditcard_2013.csv --out outputs/quick --quick
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccfd.config import DEFAULT_TRIALS, default_config  # noqa: E402
from ccfd.experiment import run_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ds1", "ds2"], required=True)
    ap.add_argument("--data", required=True, help="path to the CSV file")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-jobs", type=int, default=None, help="threads (default: -1 for DS1, 1 for DS2)")
    ap.add_argument("--trials-scale", type=float, default=1.0, help="multiply every Optuna budget")
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 trials per learner, light SHAP")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--subsample", type=float, default=None,
                    help="smoke tests only: fraction of legitimate rows to keep")
    ap.add_argument("--fix-catboost-eval", action="store_true",
                    help="disable CatBoost best-iteration selection on CV evaluation sets")
    ap.add_argument("--dedup", action="store_true", help="drop exact duplicate rows before splitting")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("shap", "optuna", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    over = {"run_shap": not args.no_shap,
            "catboost_disable_best_model": args.fix_catboost_eval,
            "dedup_before_split": args.dedup}
    trials = {k: max(1, int(round(v * args.trials_scale))) for k, v in DEFAULT_TRIALS.items()}
    if args.quick:
        trials = {k: 3 for k in DEFAULT_TRIALS}
        over.update(n_startup_trials=2, svm_eval_max=50, svm_bg_max=20, shap_nsamples=50,
                    tree_eval_max=500, tree_bg_max=200, ens_eval_max=50, ens_kmeans=10, ens_nsamples=30)
    over["n_trials"] = trials
    if args.subsample is not None:
        over["legit_subsample"] = args.subsample
    if args.n_jobs is not None:
        over["n_jobs"] = args.n_jobs
    cfg = default_config(args.dataset, args.data, args.out, **over)
    res = run_experiment(cfg)
    cols = ["Rank_AP", "Configuration", "AP", "CV_AP_SD", "ROC-AUC", "Threshold",
            "Precision", "Recall", "F2", "MCC", "TP", "FN", "FP"]
    print(res["results"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
