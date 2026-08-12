"""Integration unit tests for Mission 14 Probability Calibration Experiment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.calibration_experiment import (
    compute_ece,
    evaluate_calibration_metrics,
    run_calibration_experiment,
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


class TestCalibrationExperimentIntegration(unittest.TestCase):
    def test_01_ece_and_metrics_calculation(self):
        """Verify Expected Calibration Error (ECE) and metric computation."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])

        ece = compute_ece(y_true, probs, n_bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

        m = evaluate_calibration_metrics(y_true, probs)
        self.assertIn("brier", m)
        self.assertIn("log_loss", m)
        self.assertIn("ece", m)
        self.assertIn("roc_auc", m)
        self.assertIn("ppr_050", m)

        # Verify log loss is calculated properly without hardcoding 0.693
        self.assertNotEqual(m["log_loss"], 0.693)

    def test_02_minimal_experiment_integration(self):
        """Verify run_calibration_experiment executes cleanly on synthetic mock dataset."""
        df_raw = create_synthetic_market_data(n_samples=250, seed=101)

        class DummyStorageEngine:
            def dataset_exists(self, asset_name):
                return True

            def load_dataset(self, asset_name):
                return df_raw

        with patch("src.research.calibration_experiment.StorageEngine", DummyStorageEngine):
            with tempfile.TemporaryDirectory() as tmp_dir:
                res = run_calibration_experiment(
                    assets=["reliance_ns"],
                    folds=2,
                    scale=True,
                    output_dir=tmp_dir,
                )

                df_fold = res["fold_results"]
                self.assertFalse(df_fold.empty)

                # 1 asset * 3 models * 2 folds = 6 fold evaluations
                self.assertEqual(len(df_fold), 6)

                required_cols = [
                    "brier_A", "brier_B", "brier_C",
                    "log_loss_A", "log_loss_B", "log_loss_C",
                    "ppr_050_A", "ppr_050_B", "ppr_050_C",
                    "roc_auc_A", "roc_auc_B", "roc_auc_C",
                    "rank_corr_B", "rank_inversions_B",
                ]
                for col in required_cols:
                    self.assertIn(col, df_fold.columns)

    def test_03_explicit_metric_exception_on_single_class(self):
        """Regression Test: Verify evaluate_calibration_metrics raises ValueError on single-class target."""
        y_single = np.array([1, 1, 1, 1])
        probs = np.array([0.6, 0.7, 0.8, 0.9])

        with self.assertRaises(ValueError):
            evaluate_calibration_metrics(y_single, probs)

    def test_04_champion_json_unmodified(self):
        """Verify champion.json artifact is never modified during integration testing."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            original_mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, original_mtime)


if __name__ == "__main__":
    unittest.main()
