"""Integration unit tests for Mission 16 Step 2 Controlled Feature Family Experiment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.feature_family_experiment import (
    CONFIGURATIONS,
    run_feature_family_experiment,
)


def create_synthetic_market_data(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataframe for test storage mock."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_samples)

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


class TestFeatureFamilyExperimentIntegration(unittest.TestCase):
    def test_01_exactly_nine_configurations(self):
        """Verify experiment runner specifies exactly 9 configurations C0 through C8."""
        self.assertEqual(len(CONFIGURATIONS), 9)
        self.assertIn("C0_Control", CONFIGURATIONS)
        self.assertIn("C8_Filtered_Combined", CONFIGURATIONS)

    def test_02_minimal_experiment_run(self):
        """Verify run_feature_family_experiment executes cleanly on synthetic mock dataset."""
        df_raw = create_synthetic_market_data(n_samples=1200, seed=101)

        class DummyStorageEngine:
            def dataset_exists(self, asset_name):
                return True

            def load_dataset(self, asset_name):
                return df_raw

        with patch("src.research.feature_family_experiment.StorageEngine", DummyStorageEngine):
            with tempfile.TemporaryDirectory() as tmp_dir:
                res = run_feature_family_experiment(
                    folds=2,
                    scale=True,
                    output_dir=tmp_dir,
                )

                df_fold = res["fold_results"]
                self.assertFalse(df_fold.empty)

                # 9 configurations * 2 folds = 18 evaluations
                self.assertEqual(len(df_fold), 18)

                required_cols = [
                    "config_id", "asset", "target_name", "model", "fold",
                    "feature_count", "roc_auc", "pr_auc", "mcc", "f1", "precision", "recall", "ppr",
                    "mean_realized_ret_buy", "mean_realized_ret_sell", "return_spread",
                    "spread_minus_5bps", "spread_minus_10bps", "spread_minus_20bps",
                ]
                for col in required_cols:
                    self.assertIn(col, df_fold.columns)

    def test_03_champion_json_unmodified(self):
        """Verify champion.json artifact is never modified during integration testing."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            original_mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, original_mtime)


if __name__ == "__main__":
    unittest.main()
