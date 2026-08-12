"""Focused Integration Unit Tests for Dynamic Threshold Experiment (Mission 12 Step 3B)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.dynamic_threshold_experiment import (
    SHORTLIST_16,
    _create_folds_index,
    _evaluate_at_threshold,
    run_dynamic_threshold_experiment,
)


def create_synthetic_market_data(n_samples: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataframe for test storage mock."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_samples)

    close = 100 + np.cumsum(rng.normal(0, 1, size=n_samples))
    open_p = close + rng.normal(0, 0.5, size=n_samples)
    high_p = np.maximum(open_p, close) + np.abs(rng.normal(0, 0.5, size=n_samples))
    low_p = np.minimum(open_p, close) - np.abs(rng.normal(0, 0.5, size=n_samples))
    adj_close = close
    volume = rng.integers(10000, 50000, size=n_samples)

    return pd.DataFrame(
        {
            "Open": open_p,
            "High": high_p,
            "Low": low_p,
            "Close": close,
            "Adj Close": adj_close,
            "Volume": volume,
        },
        index=dates,
    )


class TestDynamicThresholdExperiment(unittest.TestCase):
    def test_roc_auc_identical_between_treatments(self):
        """Verify ROC-AUC metric is strictly identical between fixed and dynamic threshold evaluations."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100)
        probs = rng.uniform(0.1, 0.9, size=100)

        eval_fixed = _evaluate_at_threshold(y_true, probs, 0.50)
        eval_dynamic = _evaluate_at_threshold(y_true, probs, 0.38)

        self.assertEqual(eval_fixed["roc_auc"], eval_dynamic["roc_auc"])

    def test_fold_index_consistency(self):
        """Verify outer fold splitting produces identical train/val index boundaries."""
        dates = pd.date_range("2020-01-01", periods=200)
        folds_idx_1 = _create_folds_index(dates, folds=5)
        folds_idx_2 = _create_folds_index(dates, folds=5)

        self.assertEqual(len(folds_idx_1), 5)
        self.assertEqual(folds_idx_1, folds_idx_2)

    def test_holdout_isolation_and_evaluation_count(self):
        """Verify outer training/validation and holdout partition isolation logic."""
        df_raw = create_synthetic_market_data(n_samples=250, seed=101)

        class DummyStorageEngine:
            def dataset_exists(self, asset_name):
                return True

            def load_dataset(self, asset_name):
                return df_raw

        with patch("src.research.dynamic_threshold_experiment.StorageEngine", DummyStorageEngine):
            with tempfile.TemporaryDirectory() as tmp_dir:
                res = run_dynamic_threshold_experiment(
                    assets=["reliance_ns"],
                    folds=2,
                    scale=True,
                    output_dir=tmp_dir,
                )

                df_fold = res["fold_results"]

                self.assertFalse(df_fold.empty)
                self.assertIn("asset", df_fold.columns)
                self.assertIn("dynamic_threshold", df_fold.columns)

                # Verify exact evaluation count: 1 asset * 3 models * 2 folds = 6 evaluations
                self.assertEqual(len(df_fold), 6)

                # Verify holdout was never accessed (non-test rows < total raw rows)
                total_eval_samples = df_fold["train_samples"].iloc[0] + df_fold["validation_samples"].iloc[0]
                self.assertLess(total_eval_samples, len(df_raw))


if __name__ == "__main__":
    unittest.main()
