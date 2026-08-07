# Baseline Analysis

## Purpose

The Baseline Analysis Framework provides a reproducible descriptive view of a
`DatasetBundle` before model selection. It is read-only: it does not alter
features, targets, splits, or trained models.

## Dataset Analysis

`DatasetAnalyzer` reports total, training, validation, and test samples;
feature count; missing values; duplicate feature rows; and target counts. This
confirms the quality and size of the data supplied by Dataset Builder.

## Class Distribution

`ClassDistributionAnalyzer` counts class zero and class one, computes their
percentages, and writes `reports/analysis/class_distribution.png`. The result
helps identify class imbalance before classifier evaluation.

## Feature Analysis

`FeatureAnalyzer` calculates mean, median, standard deviation, minimum,
maximum, variance, skewness, and kurtosis for every feature across the full
chronological dataset bundle.

## Correlation Analysis

`CorrelationAnalyzer` computes a Pearson correlation matrix and each feature's
correlation with the target. It saves `correlation_matrix.csv`,
`target_correlation.csv`, and `correlation_heatmap.png` in `reports/analysis/`.
Positive and negative target correlations are ranked separately.

## Baseline Report

`ReportGenerator` coordinates all analyses and writes
`reports/analysis/baseline_report.md`. The report contains dataset statistics,
target balance, correlation outputs, strongest and weakest target-correlated
features, and data-driven modelling recommendations.
