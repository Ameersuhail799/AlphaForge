"""Tests for Multi-Asset Feature Research (Mission 11).

Includes synthetic unit tests for leak safety, scaler isolation, holdout segregation,
and multi-asset execution.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.dataset.scaler import FeatureScaler
from src.research.multi_asset_feature_research import (
    DEFAULT_ASSETS,
    RAW_PRICE_LEVELS,
    SHORTLIST_16,
    _create_folds_index,
    compute_feature_rank_stability,
    compute_paired_generalization,
    run_multi_asset_feature_research,
)


def test_shortlist_16_definition():
    """Verify SHORTLIST_16 feature count and contents."""
    assert len(SHORTLIST_16) == 16
    assert "GAP_PCT" in SHORTLIST_16
    assert "ROC_12" in SHORTLIST_16
    assert "ROLLING_STD_20" in SHORTLIST_16
    assert "VOLUME_RATIO" in SHORTLIST_16
    for price_col in RAW_PRICE_LEVELS:
        assert price_col not in SHORTLIST_16


def test_folds_index_expanding_window():
    """Verify chronological expanding-window fold generation."""
    idx = pd.date_range("2020-01-01", periods=100)
    folds_pos = _create_folds_index(idx, folds=5)
    assert len(folds_pos) == 5

    prev_train_end = -1
    for train_end, val_end in folds_pos:
        assert train_end > prev_train_end
        assert val_end > train_end
        prev_train_end = train_end


def test_scaler_isolation_no_leakage():
    """Verify FeatureScaler fits strictly on training slice and does not see validation/holdout data."""
    X_train = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    X_val = pd.DataFrame({"feat1": [10.0, 20.0, 30.0]})

    scaler = FeatureScaler(scale=True)
    X_train_scaled = scaler.fit_transform_train(X_train)

    assert scaler._mean is not None
    train_mean = float(scaler._mean["feat1"])
    assert np.isclose(train_mean, 3.0)

    # Validation transform should use train mean (3.0), not val mean (20.0)
    X_val_scaled = scaler.transform(X_val)
    val_unscaled_back = X_val_scaled["feat1"].values * float(scaler._scale["feat1"]) + train_mean
    assert np.allclose(val_unscaled_back, X_val["feat1"].values)


def test_holdout_segregation():
    """Verify final 15% holdout test partition is strictly excluded from cross-validation folds."""
    total_rows = 100
    test_size = max(1, int(total_rows * 0.15))
    non_test_rows = total_rows - test_size

    idx = pd.date_range("2020-01-01", periods=total_rows)
    non_test_idx = idx[:-test_size]

    folds_pos = _create_folds_index(non_test_idx, folds=5)
    max_val_end = max(val_end for _, val_end in folds_pos)

    # Max validation index must remain inside non_test boundary (non_test_rows - 1)
    assert max_val_end < non_test_rows
    assert max_val_end < total_rows - test_size


def test_rank_stability_computation():
    """Verify Spearman rank stability calculation logic."""
    df_fi = pd.DataFrame([
        {"asset": "asset1", "config": "SHORTLIST_16", "model": "rf", "fold": 1, "feature": "f1", "importance": 0.8},
        {"asset": "asset1", "config": "SHORTLIST_16", "model": "rf", "fold": 1, "feature": "f2", "importance": 0.2},
        {"asset": "asset1", "config": "SHORTLIST_16", "model": "rf", "fold": 2, "feature": "f1", "importance": 0.9},
        {"asset": "asset1", "config": "SHORTLIST_16", "model": "rf", "fold": 2, "feature": "f2", "importance": 0.1},
    ])
    res = compute_feature_rank_stability(df_fi)
    assert not res.empty
    assert "spearman_rank_correlation" in res.columns
    assert np.isclose(res["spearman_rank_correlation"].iloc[0], 1.0)


def test_paired_generalization_computation():
    """Verify paired comparison computation between SHORTLIST_16 and ALL_32."""
    df_group = pd.DataFrame([
        {"asset": "a1", "feature_config": "SHORTLIST_16", "model": "rf", "fold": 1, "roc_auc": 0.55, "f1": 0.52, "is_model_collapse": False},
        {"asset": "a1", "feature_config": "ALL_32", "model": "rf", "fold": 1, "roc_auc": 0.50, "f1": 0.48, "is_model_collapse": True},
        {"asset": "a1", "feature_config": "SHORTLIST_16", "model": "rf", "fold": 2, "roc_auc": 0.54, "f1": 0.51, "is_model_collapse": False},
        {"asset": "a1", "feature_config": "ALL_32", "model": "rf", "fold": 2, "roc_auc": 0.49, "f1": 0.45, "is_model_collapse": True},
    ])
    res = compute_paired_generalization(df_group)
    assert not res.empty
    row = res.iloc[0]
    assert row["model"] == "rf"
    assert row["mean_auc_diff"] > 0
    assert row["win_rate_auc"] == 1.0
    assert row["model_collapse_all_32"] == 2
    assert row["model_collapse_shortlist_16"] == 0


def test_multi_asset_feature_research_integration():
    """Integration test running multi-asset feature research across available assets."""
    res = run_multi_asset_feature_research(
        assets=DEFAULT_ASSETS,
        folds=3,
        scale=True,
    )

    assert "group" in res
    assert "feature_stability" in res
    assert "generalization" in res

    assert os.path.exists("reports/research/multi_asset_group_comparison.csv")
    assert os.path.exists("reports/research/multi_asset_feature_stability.csv")
    assert os.path.exists("reports/research/feature_subset_generalization.csv")
    assert os.path.exists("reports/research/MULTI_ASSET_FEATURE_ROBUSTNESS_REPORT.md")

    # Verify CSV files are non-empty
    assert os.path.getsize("reports/research/multi_asset_group_comparison.csv") > 0
    assert os.path.getsize("reports/research/multi_asset_feature_stability.csv") > 0
    assert os.path.getsize("reports/research/feature_subset_generalization.csv") > 0
    assert os.path.getsize("reports/research/MULTI_ASSET_FEATURE_ROBUSTNESS_REPORT.md") > 0

    print("MULTI-ASSET FEATURE RESEARCH TESTS PASSED")


if __name__ == "__main__":
    test_shortlist_16_definition()
    test_folds_index_expanding_window()
    test_scaler_isolation_no_leakage()
    test_holdout_segregation()
    test_rank_stability_computation()
    test_paired_generalization_computation()
    test_multi_asset_feature_research_integration()
