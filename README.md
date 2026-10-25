# Leakage-aware credit card fraud detection: optimized, ensemble and SHAP-explained classifiers

Code for the SN Computer Science article *Do Ensembles Pay Off? A Leakage-Aware Comparison of Optimized, Ensemble and SHAP-Explained Classifiers for Credit Card Fraud Detection Across Two Datasets*.

It reimplements the two dissertation notebooks (`CC_code_2013_ap.ipynb`, `CC_code_2023_ap.ipynb`) as one package, with the same seeds, split, search spaces, thresholds, ensembles and SHAP settings. It also computes the extra analyses reported in the paper and exports the data behind the LaTeX figures.

## What the pipeline does

For each dataset the pipeline runs these stages in order:

1. Load the CSV and cast the types, dropping the DS2 `id` column.
2. Report data quality: missing values, duplicates and 1.5-IQR outliers.
3. For DS2 only, match the fraud prevalence to 0.172% before splitting.
4. Make a stratified 80/20 split with seed 42.
5. Tune each of the six learners with Optuna/TPE on mean 5-fold CV average precision. All six studies share one seeded sampler, as in the notebooks.
6. Select a threshold per fold by maximizing F2, then average the fold thresholds.
7. Refit each learner on the whole training partition and score the holdout once.
8. Build four ensembles from out-of-fold scores: mean, AP-weighted mean, majority vote (3 of 6) and stacking with an LR meta-learner.
9. Compute SHAP for the base learners, and for the explanatory proxies of the ensembles.

Feature construction, scaling and SMOTE are refitted inside every training fold. Validation folds and the holdout are never resampled. Thresholds and the stacking meta-learner come from training data only.

| Learner | Scaling | Imbalance handling | Trials | SHAP explainer |
|---|---|---|---|---|
| Logistic regression | min-max | none (threshold only) | 80 | Linear |
| SVM (RBF) | min-max | none (threshold only) | 40 | Kernel (decision function) |
| Random forest | none | class_weight=balanced | 60 | Kernel (log-odds) |
| XGBoost | none | SMOTE in training folds | 60 | Tree |
| LightGBM | none | SMOTE in training folds | 60 | Tree |
| CatBoost | none | auto_class_weights=Balanced | 60 | Tree |

## Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The notebooks ran on Python 3.8. The package has also been tested on Python 3.13 with current library versions. Results can shift slightly across library versions, mainly SVM probabilities and SHAP sampling.

## Data

Download the two public datasets:

- DS1: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (`creditcard.csv`)
- DS2: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023 (`creditcard_2023.csv`)

In the dissertation repository these files are stored with Git LFS. Run `git lfs pull` there, or download them from Kaggle. The loader refuses LFS pointer files with a clear message.

## Running

Each command below runs one stage of the workflow.

Run the full DS1 experiment (several hours, dominated by SVM and boosting tuning):

```
python scripts/run_experiment.py --dataset ds1 --data data/creditcard_2013.csv --out outputs/ds1
```

Run the full DS2 experiment (prevalence matched to 0.172%):

```
python scripts/run_experiment.py --dataset ds2 --data data/creditcard_2023.csv --out outputs/ds2
```

Compare the two runs (rank correlation, top-k frequency, Jaccard agreement):

```
python scripts/compare_datasets.py --ds1 outputs/ds1 --ds2 outputs/ds2 --out outputs/compare
```

Rebuild Table 8 of the paper from the published precision and recall. This needs no data:

```
python scripts/paper_operating_points.py
```

Useful flags for `run_experiment.py`:

- `--quick`: 3 trials per learner and light SHAP. This is a smoke test only, not for reporting.
- `--subsample 0.1`: keeps every fraud and 10% of legitimate rows. Use it only with `--quick` for smoke tests.
- `--trials-scale 0.5`: halve every Optuna budget.
- `--no-shap`: skip explanations.
- `--n-jobs N`: thread count. The defaults follow the notebooks (-1 for DS1, 1 for DS2).
- `--fix-catboost-eval`: disables CatBoost's best-iteration selection on the CV evaluation set, which the paper lists as a threat to validity.
- `--dedup`: drops exact duplicate rows before splitting, the other DS1 caveat.

Both corrections are off by default so that the paper's numbers are reproduced.

For a dry run without the real data:

```
python scripts/make_synthetic.py --out data_synthetic
python scripts/run_experiment.py --dataset ds1 --data data_synthetic/synthetic_ds1.csv --out outputs/smoke --quick --subsample 0.3
```

## Outputs (per run)

| File | Content | Paper |
|---|---|---|
| `data_summary.json`, `outliers_per_feature.csv` | quality checks, split sizes | Table 2, Sec. 3.3 |
| `optuna_trials.csv`, `optimization_summary.csv`, `best_params.json` | every trial, best/median/range/TPE gain, selected values | Fig. 4, Table 5, Table A1 |
| `holdout_results.csv` | AP, CV SD, ROC-AUC, threshold, P, R, F2, MCC, TP/FP/FN/TN, alerts per 10k, FPR, Wilson CIs | Tables 6-8, Figs. 7-8 |
| `threshold_stability.csv` | per-fold F2 thresholds, mean and SD | Fig. 9 |
| `cv_vs_holdout.csv` | CV AP against holdout AP | Sec. 4.6 |
| `ensemble_internals.csv` | AP weights and stacking-proxy coefficients | Table 9 |
| `shap/base_*.csv`, `shap/ensemble_*.csv` | top-15 SHAP per learner; ensemble contributions | Figs. 10-12, Tables 10, B2-B3 |
| `curves/pr_*.csv`, `curves/roc_*.csv` | exact holdout PR and ROC curves | Figs. 5-6 |
| `latex/pgfplots_coordinates.json` | ready-to-paste `\addplot coordinates` strings (PR, ROC, convergence) | Figs. 4-6 |
| `holdout_scores.csv`, `oof_scores.csv`, `models/` | raw scores and fitted models | |

`compare_datasets.py` writes `cross_dataset_ap.csv` (Table 11, Fig. 14), `*_top5_frequency.csv`, `*_jaccard_top{5,10}.csv` (Fig. 13), `*_rank_matrix.csv` (Fig. 12) and `cross_dataset_stats.json`.

### Replacing the traced curves in the LaTeX source

In the manuscript, the PR and ROC curves in Figs. 5 and 6 were traced from the thesis plots. After a full run, take the strings from `outputs/ds1/latex/pgfplots_coordinates.json` and `outputs/ds2/latex/pgfplots_coordinates.json`. The keys are `pr_LightGBM`, `roc_XGBoost` and so on. Paste each string between the braces of the matching `\addplot ... coordinates {...}` line. The curve keys use the short names LR, SVM, RF, XGBoost, LightGBM and CatBoost.

## Package layout

```
ccfd/
  config.py        settings, model plan, trial budgets (paper defaults)
  data.py          loading, quality report, prevalence matching, stratified split
  features.py      log amount, cyclical hour, conditional scaling, fold-local SMOTE
  models.py        model factory, Optuna search spaces, fold and final fitting
  optimization.py  Optuna studies, trial tables, CV thresholds
  metrics.py       AP/ROC/P/R/F2/MCC, F2 threshold, confusion counts, Wilson CIs
  ensembles.py     OOF scores, mean/weighted/majority vote/stacking
  explain.py       SHAP for base learners and ensemble proxies
  curves.py        exact PR/ROC export and pgfplots coordinates
  experiment.py    end-to-end run for one dataset
  compare.py       cross-dataset ranking and SHAP agreement
scripts/           command-line entry points
tests/             unit tests (python -m pytest -q tests)
```

## Notes

- Ensemble SHAP values describe explanatory proxies on quantile-normalized scores. They are not exact decompositions of the predictive ensembles.
- Majority vote AP and ROC-AUC are computed from binary decisions and are not comparable with score-based values.
- Wall-clock times depend heavily on thread settings and hardware, and are not a result of the study.
