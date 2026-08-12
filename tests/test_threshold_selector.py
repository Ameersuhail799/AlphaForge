"""Unit tests for Leak-Safe Dynamic Threshold Selector (Mission 12 Step 3A)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.threshold_selector import select_dynamic_threshold


def create_synthetic_data(n_samples: int = 300, seed: int = 42):
    """Generate synthetic chronological market feature data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_samples)

    df = pd.DataFrame(
        {
            "GAP_PCT": rng.normal(0, 0.01, size=n_samples),
            "OPEN_CLOSE_PCT": rng.normal(0, 0.01, size=n_samples),
            "HIGH_LOW_PCT": rng.uniform(0.01, 0.05, size=n_samples),
            "BODY_SIZE": rng.uniform(0.001, 0.03, size=n_samples),
            "UPPER_WICK": rng.uniform(0.001, 0.02, size=n_samples),
            "LOWER_WICK": rng.uniform(0.001, 0.02, size=n_samples),
            "ROC_12": rng.normal(0, 0.02, size=n_samples),
            "RSI_14": rng.uniform(30, 70, size=n_samples),
            "PRICE_CHANGE_PCT": rng.normal(0, 0.02, size=n_samples),
            "DAILY_RETURN": rng.normal(0, 0.01, size=n_samples),
            "ROLLING_STD_20": rng.uniform(0.01, 0.03, size=n_samples),
            "HIST_VOL_20": rng.uniform(0.1, 0.3, size=n_samples),
            "ATR_14": rng.uniform(1.0, 5.0, size=n_samples),
            "DAILY_RANGE_PCT": rng.uniform(0.01, 0.04, size=n_samples),
            "VOLUME_RATIO": rng.uniform(0.5, 2.0, size=n_samples),
            "VOLUME_CHANGE_PCT": rng.normal(0, 0.1, size=n_samples),
        },
        index=dates,
    )

    # Synthetic signal for target
    signal = 2.0 * df["ROC_12"] + 1.5 * df["GAP_PCT"] - 0.02 * df["RSI_14"]
    target = (signal > signal.median()).astype(int)

    return df, pd.Series(target, index=dates)


def test_threshold_selection_bounds_and_feasibility():
    """Test selected threshold is strictly within [0.20, 0.80] and meets Recall >= 0.35 when feasible."""
    X_train, y_train = create_synthetic_data(300, seed=42)
    cols = list(X_train.columns)

    tau, meta = select_dynamic_threshold(
        outer_X_train=X_train,
        outer_y_train=y_train,
        model_name="logistic_regression",
        feature_cols=cols,
        inner_folds=3,
        min_recall_floor=0.35,
    )

    assert 0.20 <= tau <= 0.80
    assert "oof_recall" in meta
    assert "oof_f1" in meta

    if meta["feasible_count"] > 0:
        assert meta["oof_recall"] >= 0.35
        assert not meta["fallback_used"]


def test_scaler_independent_inner_folds():
    """Verify FeatureScaler is independently fitted inside each inner fold."""
    X_train, y_train = create_synthetic_data(300, seed=123)
    cols = list(X_train.columns)

    tau, meta = select_dynamic_threshold(
        outer_X_train=X_train,
        outer_y_train=y_train,
        model_name="random_forest",
        feature_cols=cols,
        inner_folds=3,
    )

    scaler_h = meta["scaler_histories"]
    assert len(scaler_h) > 0

    # Ensure mean values change across expanding inner fold training windows
    means_f1 = scaler_h[0]["mean"]
    means_f2 = scaler_h[1]["mean"]
    assert means_f1 != means_f2


def test_fallback_when_no_feasible_threshold():
    """Verify fallback behavior when recall floor cannot be satisfied."""
    X_train, y_train = create_synthetic_data(200, seed=999)
    cols = list(X_train.columns)

    # Force an unachievable recall floor (1.01) to trigger fallback
    tau, meta = select_dynamic_threshold(
        outer_X_train=X_train,
        outer_y_train=y_train,
        model_name="logistic_regression",
        feature_cols=cols,
        inner_folds=3,
        min_recall_floor=1.01,
    )

    assert 0.20 <= tau <= 0.80
    assert meta["fallback_used"]
    assert meta["feasible_count"] == 0


def test_deterministic_tie_breaking():
    """Verify threshold selection is completely deterministic across multiple runs."""
    X_train, y_train = create_synthetic_data(250, seed=777)
    cols = list(X_train.columns)

    tau1, _ = select_dynamic_threshold(
        outer_X_train=X_train,
        outer_y_train=y_train,
        model_name="xgboost",
        feature_cols=cols,
        inner_folds=3,
    )

    tau2, _ = select_dynamic_threshold(
        outer_X_train=X_train,
        outer_y_train=y_train,
        model_name="xgboost",
        feature_cols=cols,
        inner_folds=3,
    )

    assert tau1 == tau2


def test_no_outer_validation_leakage():
    """Verify helper operates exclusively on outer_X_train and outer_y_train without outer_X_val."""
    X_full, y_full = create_synthetic_data(400, seed=101)

    # Split into outer train (300) and outer val (100)
    outer_X_train = X_full.iloc[:300]
    outer_y_train = y_full.iloc[:300]

    outer_X_val = X_full.iloc[300:]
    outer_y_val = y_full.iloc[300:]

    tau, meta = select_dynamic_threshold(
        outer_X_train=outer_X_train,
        outer_y_train=outer_y_train,
        model_name="logistic_regression",
        feature_cols=list(outer_X_train.columns),
        inner_folds=3,
    )

    # Confirm total OOF samples used for threshold tuning < outer_X_train length (never includes outer_X_val)
    assert meta["oof_samples"] <= len(outer_X_train)
    assert meta["oof_samples"] < len(X_full)


if __name__ == "__main__":
    test_threshold_selection_bounds_and_feasibility()
    test_scaler_independent_inner_folds()
    test_fallback_when_no_feasible_threshold()
    test_deterministic_tie_breaking()
    test_no_outer_validation_leakage()
    print("THRESHOLD SELECTOR UNIT TESTS PASSED")
