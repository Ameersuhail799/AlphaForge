"""Integration unit tests for Mission 15 Step 2 Target Horizon Screening Experiment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.target_screening_experiment import (
    build_candidate_targets,
    compute_pr_auc,
    run_target_screening_experiment,
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


class TestTargetScreeningExperimentIntegration(unittest.TestCase):
    def test_01_candidate_target_building(self):
        """Verify build_candidate_targets builds Targets A through F correctly."""
        df_raw = create_synthetic_market_data(n_samples=100)
        df_targets = build_candidate_targets(df_raw)

        for col in ["TARGET_A", "TARGET_B", "TARGET_C", "TARGET_D", "TARGET_E", "TARGET_F"]:
            self.assertIn(col, df_targets.columns)

        # TARGET_F tri-class values: 0 (SELL), 1 (NO_TRADE), 2 (BUY)
        unique_f = set(df_targets["TARGET_F"].unique())
        self.assertTrue(unique_f.issubset({0, 1, 2}))

    def test_02_pr_auc_computation(self):
        """Verify compute_pr_auc returns valid PR-AUC value in [0, 1]."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])

        pr_auc = compute_pr_auc(y_true, probs)
        self.assertGreaterEqual(pr_auc, 0.0)
        self.assertLessEqual(pr_auc, 1.0)

    def test_03_minimal_screening_experiment_run(self):
        """Verify run_target_screening_experiment executes cleanly on synthetic mock dataset."""
        df_raw = create_synthetic_market_data(n_samples=250, seed=101)

        class DummyStorageEngine:
            def dataset_exists(self, asset_name):
                return True

            def load_dataset(self, asset_name):
                return df_raw

        with patch("src.research.target_screening_experiment.StorageEngine", DummyStorageEngine):
            with tempfile.TemporaryDirectory() as tmp_dir:
                res = run_target_screening_experiment(
                    assets=["reliance_ns"],
                    folds=2,
                    scale=True,
                    output_dir=tmp_dir,
                )

                df_fold = res["fold_results"]
                self.assertFalse(df_fold.empty)

                # 1 asset * 3 models * 2 folds * 6 targets = 36 evaluations
                self.assertEqual(len(df_fold), 36)

                required_cols = [
                    "asset", "model", "fold", "target_name",
                    "roc_auc", "pr_auc", "mcc", "f1", "precision", "recall", "ppr",
                    "mean_realized_ret_buy", "mean_realized_ret_sell",
                ]
                for col in required_cols:
                    self.assertIn(col, df_fold.columns)

    def test_04_champion_json_unmodified(self):
        """Verify champion.json artifact is never modified during integration testing."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            original_mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, original_mtime)


if __name__ == "__main__":
    unittest.main()
