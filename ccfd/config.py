"""Experiment configuration.

Defaults reproduce the settings used in the paper and in the original
dissertation notebooks (CC_code_2013_ap.ipynb / CC_code_2023_ap.ipynb).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

# Model plan: (learner, scaler, imbalance strategy). Order matters, because
# the Optuna TPE sampler is shared across studies (as in the notebooks), so
# changing the order changes the random stream seen by each study.
MODELS_PLAN = [
    ("Logistic Regression", "MinMaxScaler", "none"),
    ("SVM (RBF)", "MinMaxScaler", "none"),
    ("Random Forest", "None (no scaling)", "class_weight"),
    ("XGBoost", "None (no scaling)", "smote"),
    ("LightGBM", "None (no scaling)", "smote"),
    ("CatBoost", "None (no scaling)", "none"),  # class weights set inside the model
]

DEFAULT_TRIALS = {
    "Logistic Regression": 80,
    "SVM (RBF)": 40,
    "Random Forest": 60,
    "XGBoost": 60,
    "LightGBM": 60,
    "CatBoost": 60,
}

# Short names used in tables and LaTeX exports.
SHORT = {
    "Logistic Regression": "LR",
    "SVM (RBF)": "SVM",
    "Random Forest": "RF",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "CatBoost": "CatBoost",
}


@dataclass
class Config:
    # Dataset identity: "ds1" (2013 European cardholders) or "ds2" (2023 dataset)
    dataset: str = "ds1"
    csv_path: str = "data/creditcard_2013.csv"
    output_dir: str = "outputs/ds1"

    # Reproducibility
    random_state: int = 42

    # Split and cross-validation
    test_size: float = 0.2
    n_splits_max: int = 5

    # DS2 prevalence matching (fraction, applied before the split). None = off.
    target_fraud_rate: Optional[float] = None

    # Optuna
    n_trials: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TRIALS))
    n_startup_trials: int = 10

    # Threads. The DS1 notebook used -1 for LR/RF/XGBoost and the DS2 notebook used 1.
    n_jobs: int = -1

    # SHAP settings (as in the notebooks)
    do_svm_shap: bool = True
    svm_bg_max: int = 200        # background samples for KernelExplainer (RF, SVM)
    svm_eval_max: int = 1000     # holdout rows explained with KernelExplainer (RF, SVM)
    shap_nsamples: int = 200     # coalition samples for KernelExplainer (RF, SVM)
    tree_eval_max: int = 5000    # holdout rows explained with TreeExplainer
    tree_bg_max: Optional[int] = None  # None = full training background (notebook behaviour)
    ens_kmeans: int = 50
    ens_eval_max: int = 500
    ens_nsamples: int = 100
    shap_top_k: int = 15

    # Optional corrections discussed in the paper (Threats to validity).
    # Off by default so that the paper's numbers are reproduced.
    catboost_disable_best_model: bool = False   # avoid best-iteration selection on the CV eval set
    dedup_before_split: bool = False            # drop exact duplicate rows before splitting

    # Skip expensive parts (useful for smoke tests)
    run_shap: bool = True
    # Smoke tests only: keep all frauds and this fraction of legitimate rows (None = all)
    legit_subsample: Optional[float] = None

    @property
    def is_ds1(self) -> bool:
        return self.dataset.lower() == "ds1"


def default_config(dataset: str, csv_path: str, output_dir: str, **overrides) -> Config:
    """Configuration matching the notebook for each dataset."""
    ds = dataset.lower()
    if ds not in {"ds1", "ds2"}:
        raise ValueError("dataset must be 'ds1' or 'ds2'")
    cfg = Config(dataset=ds, csv_path=csv_path, output_dir=output_dir)
    if ds == "ds2":
        cfg.target_fraud_rate = 0.00172
        cfg.n_jobs = 1
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Unknown config field: {k}")
        setattr(cfg, k, v)
    return cfg
