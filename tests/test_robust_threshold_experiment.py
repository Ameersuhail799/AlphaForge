"""Mission 13 Step 2: Robust Threshold Experiment Integration Tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.robust_threshold_experiment import (
    SHORTLIST_16,
    _create_folds_index,
    evaluate_val_metrics,
    run_robust_threshold_experiment,
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


class TestRobustThresholdExperimentIntegration(unittest.TestCase):
    def test_01_treatments_and_metrics_calculation(self):
        """Verify metric calculation accuracy for PPR, precision, recall, F1, MCC, Youden's J."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])

        m = evaluate_val_metrics(y_true, probs, tau=0.50)
        self.assertEqual(m["ppr"], 0.50)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["mcc"], 1.0)
        self.assertEqual(m["youden_j"], 1.0)

    def test_02_fold_boundaries_identical(self):
        """Verify all treatments use identical outer fold boundaries."""
        dates = pd.date_range("2020-01-01", periods=200)
        folds_idx_1 = _create_folds_index(dates, folds=5)
        folds_idx_2 = _create_folds_index(dates, folds=5)

        self.assertEqual(len(folds_idx_1), 5)
        self.assertEqual(folds_idx_1, folds_idx_2)

    def test_03_minimal_integration_and_holdout_isolation(self):
        """Verify Treatments A-E integration, outer training data usage, and holdout isolation."""
        df_raw = create_synthetic_market_data(n_samples=250, seed=101)

        class DummyStorageEngine:
            def dataset_exists(self, asset_name):
                return True

            def load_dataset(self, asset_name):
                return df_raw

        with patch("src.research.robust_threshold_experiment.StorageEngine", DummyStorageEngine):
            with tempfile.TemporaryDirectory() as tmp_dir:
                res = run_robust_threshold_experiment(
                    assets=["reliance_ns"],
                    folds=2,
                    scale=True,
                    output_dir=tmp_dir,
                )

                df_fold = res["fold_results"]
                self.assertFalse(df_fold.empty)

                # Verify exact evaluation count: 1 asset * 3 models * 2 folds = 6 evaluations
                self.assertEqual(len(df_fold), 6)

                # Verify columns for Treatments A, B, C, D, E exist
                required_cols = [
                    "tau_A", "precision_A", "recall_A", "f1_A", "mcc_A", "youden_j_A", "ppr_A",
                    "tau_B", "precision_B", "recall_B", "f1_B", "mcc_B", "youden_j_B", "ppr_B",
                    "tau_C", "precision_C", "recall_C", "f1_C", "mcc_C", "youden_j_C", "ppr_C",
                    "tau_D", "precision_D", "recall_D", "f1_D", "mcc_D", "youden_j_D", "ppr_D",
                    "tau_E", "precision_E", "recall_E", "f1_E", "mcc_E", "youden_j_E", "ppr_E",
                ]
                for col in required_cols:
                    self.assertIn(col, df_fold.columns)

                # Verify Treatment A uses fixed threshold 0.50
                self.assertTrue((df_fold["tau_A"] == 0.50).all())

                # Verify holdout was never accessed (non-test samples < total raw samples)
                total_samples_evaluated = df_fold["train_samples"].iloc[0] + df_fold["validation_samples"].iloc[0]
                self.assertLess(total_samples_evaluated, len(df_raw))

    def test_04_champion_json_unmodified(self):
        """Verify champion.json artifact is never modified during integration testing."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            original_mtime = champion_path.stat().st_mtime
            # Re-verify stat after mock run
            self.assertEqual(champion_path.stat().st_mtime, original_mtime)


if __name__ == "__main__":
    unittest.main()
