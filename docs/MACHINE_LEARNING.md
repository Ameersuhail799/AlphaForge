# Machine Learning

## Architecture

The Machine Learning layer consumes `DatasetBundle` objects exclusively. Raw
market data and feature-engineering DataFrames are not accepted by models,
trainers, predictors, or evaluators. This preserves a single, controlled
boundary between dataset construction and modelling.

`BaseModel` defines the common model interface. `LogisticRegressionModel` is
the first production implementation and wraps
`sklearn.linear_model.LogisticRegression`.

## Trainer and Persistence

`Trainer` fits a `BaseModel` with the training partition in a `DatasetBundle`.
It stores the trained model in `models/trained/` as a `.joblib` artifact and
provides `load_model()` for restoring that artifact. The trainer also exports
feature-importance and experiment-report artifacts.

## Predictor

`Predictor` accepts a trained `BaseModel` and a `DatasetBundle`. It returns
indexed test-set class predictions and positive-class probabilities. This keeps
prediction data aligned with the chronological test index.

## Evaluator

`Evaluator` compares predictions with `DatasetBundle.y_test` and produces
accuracy, precision, recall, F1 score, ROC-AUC, a confusion matrix, and a
classification report. ROC-AUC is recorded as `null` when the test partition
does not contain both target classes.

## Feature Importance

For Logistic Regression, coefficient importance is the absolute coefficient
magnitude. The report includes the signed coefficient and absolute importance,
is sorted descending by importance, and is stored in
`reports/feature_importance/`.

## Experiment Tracking

Each experiment JSON file is saved in `reports/experiments/`. It records a
unique experiment ID, UTC timestamp, model name and parameters, feature count,
partition sample counts, and all evaluation outputs. These records make model
results traceable without embedding experiment logic in a model class.
