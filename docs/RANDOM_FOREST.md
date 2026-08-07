# Random Forest and Model Registry

## Model Registry

`ModelRegistry` centralizes model registration and construction. It
pre-registers `logistic_regression` and `random_forest`, prevents duplicate
names, and lists available model identifiers. Future model factories, including
XGBoost, LightGBM, and CatBoost, can be registered without changing trainer,
predictor, or evaluator behavior.

## Random Forest

`RandomForestModel` wraps `sklearn.ensemble.RandomForestClassifier` and accepts
only a `DatasetBundle`. Its defaults are 200 estimators, unlimited tree depth,
random state 42, and `n_jobs=-1`. Training continues to be centralized in
`Trainer`; prediction and evaluation use the existing `Predictor` and
`Evaluator` interfaces.

## Feature Importance

Random Forest importance is the estimator's impurity-based feature importance.
`Trainer.export_feature_importance()` writes descending importance to
`reports/feature_importance/random_forest_feature_importance.csv`.

## Model Comparison

After an experiment is saved, `Trainer` updates
`reports/model_comparison.csv`. The report has model, accuracy, precision,
recall, F1, ROC-AUC, training time, and prediction time columns. It seeds from
existing experiment JSON files, preserving Logistic Regression results when a
Random Forest result is added.
