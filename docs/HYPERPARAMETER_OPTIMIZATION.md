# Hyperparameter Optimization

The hyperparameter optimization framework in AlphaForge is designed to be reusable for future registered models.

## Optimizer

`HyperparameterOptimizer` accepts a registered model name and a `DatasetBundle`, then runs `RandomizedSearchCV` with `TimeSeriesSplit(n_splits=5)`.

The optimizer returns:

- the best trained model,
- the best parameter combination,
- the full optimization history,
- the optimization report metadata.

## Parameter Space

Model search spaces are centralized in `ParameterSpace`.

Currently supported:

- Random Forest

## Optimization Report

Each optimization run writes a JSON report under `reports/optimization/` with:

- best parameters,
- best score,
- all evaluated combinations,
- execution time.

## Champion Update

When optimized Random Forest performance improves the current champion, the framework refreshes the champion and leaderboard artifacts using the existing champion system.

## Ranking Notes

The search is chronological and non-shuffled. This avoids leakage and keeps evaluation aligned with time-series data.