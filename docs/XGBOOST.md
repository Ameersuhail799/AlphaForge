# XGBoost Integration

## What is XGBoost

XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements gradient boosted decision trees intended for speed and performance.

## Why AlphaForge uses XGBoost

- Strong baseline for tabular classification tasks.
- Efficient CPU implementation with feature importance and good generalization.
- Integrates with the existing AlphaForge training/evaluation pipeline.

## How it differs from Random Forest

- Random Forest is an ensemble of independently-trained decision trees (bagging).
- XGBoost builds trees sequentially where each tree corrects errors of previous ones (boosting).
- XGBoost typically yields better performance on structured/tabular data when tuned, but is more sensitive to hyperparameters.

## Model parameters

Default conservative parameters used in AlphaForge:

- `n_estimators=200`
- `max_depth=4`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `min_child_weight=3`
- `random_state=42`
- `n_jobs=-1`
- `eval_metric="logloss"`

These defaults are intentionally conservative for initial experiments.

## Feature importance

Feature importance is generated from XGBoost's `feature_importances_` attribute and exported to `reports/feature_importance/xgboost_feature_importance.csv`.

## Validation methodology

AlphaForge uses leakage-safe chronological splits:

- Hyperparameter search: `TRAIN` with `TimeSeriesSplit` cross-validation only.
- Selection: `VALIDATION` — the validation partition decides champion and hyperparameter selection.
- Final model: retrain on `TRAIN + VALIDATION` and evaluate once on `TEST` for the final experiment result.

This prevents the `TEST` partition from influencing model or hyperparameter selection.

## Champion methodology

Champion selection follows the existing AlphaForge rules (rank by):

1. ROC-AUC
2. F1
3. Accuracy

When new models (including XGBoost) are evaluated, the system will only replace the champion if the new model improves on these metrics on the `VALIDATION` partition.

## Data leakage prevention

- The `TEST` set is never used to select hyperparameters or to choose the champion.
- All selection decisions are made using `VALIDATION` metrics only.
- Final `TEST` evaluation is performed only after locking the selected model and retraining on `TRAIN + VALIDATION`.

## Current benchmark comparison

- The current production champion is `random_forest`.
- Synthetic optimization tests may show perfect scores; these are synthetic and not representative of live market performance.

---

Please refer to the code in `src/models/xgboost_model.py` for implementation details and `tests/test_xgboost.py` for an integration test example.
