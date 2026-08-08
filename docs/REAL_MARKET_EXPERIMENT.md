**Real Market Experiment**

**Purpose**: Run a fair, leakage-free comparison of `logistic_regression`, `random_forest`, and `xgboost` on the stored `RELIANCE_NS` dataset.

**Protocol**:

- Load `RELIANCE_NS` using `StorageEngine.load_dataset("RELIANCE_NS")`.
- Apply `FeaturePipeline()` and `DatasetBuilder(scale=True)` to produce a chronological `DatasetBundle` with `TRAIN`, `VALID`, and `TEST` splits.
- For each candidate model:
  - Train only on `TRAIN` and evaluate on `VALID`.
  - Collect validation metrics: ROC-AUC, F1, Accuracy, Precision, Recall.
- Select champion using VALIDATION-only metrics ordered by ROC-AUC → F1 → Accuracy.
- LOCK selection. Retrain ALL CANDIDATE MODELS (logistic_regression, random_forest, xgboost) on `TRAIN+VALID` and evaluate each exactly once on the untouched `TEST` set.
- Save model comparison CSV and JSON into `reports/` and feature importances into `reports/feature_importance/`.
- Save model comparison CSV and JSON into `reports/` and feature importances into `reports/feature_importance/`.

**Important**: Do NOT modify `reports/champion.json` or any production artifacts. This experiment writes separate research artifacts only.

**Reproducibility**: Ensure `xgboost` is installed (see `requirements-xgboost.txt`).

**Limitations**: This is a single historical backtest on `RELIANCE_NS`. Results may not generalize.
