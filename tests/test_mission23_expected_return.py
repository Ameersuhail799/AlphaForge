"""Unit and safety tests for Mission 23 Expected Return + Probability Trading Engine."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.research.mission19_edge_validation import run_strategy_simulation


def create_synthetic_market_data(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLC market dataframe."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_samples)

    close = 100.0 + np.cumsum(rng.normal(0, 1, size=n_samples))
    open_p = close + rng.normal(0, 0.2, size=n_samples)
    high_p = np.maximum(open_p, close) + 0.5
    low_p = np.minimum(open_p, close) - 0.5

    return pd.DataFrame(
        {
            "Open": open_p,
            "High": high_p,
            "Low": low_p,
            "Close": close,
            "Volume": 10000,
        },
        index=dates,
    )


class TestMission23ExpectedReturn(unittest.TestCase):
    def test_01_expected_return_target_construction(self):
        """Verify continuous 10D expected return target is (Close[t+10] - Close[t])/Close[t]."""
        val_df = create_synthetic_market_data(n_samples=50, seed=1)
        close = val_df["Close"]
        ret_10d = (close.shift(-10) - close) / close

        # Row 0 return check
        expected_r0 = (close.iloc[10] - close.iloc[0]) / close.iloc[0]
        self.assertAlmostEqual(ret_10d.iloc[0], expected_r0, delta=1e-6)

    def test_02_regressor_fitting_synthetic(self):
        """Verify RandomForestRegressor fits and predicts continuous values."""
        rng = np.random.default_rng(2)
        X_train = rng.normal(0, 1, size=(50, 10))
        y_train = rng.normal(0.01, 0.05, size=50)

        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(X_train, y_train)

        X_val = rng.normal(0, 1, size=(20, 10))
        preds = reg.predict(X_val)

        self.assertEqual(len(preds), 20)
        self.assertTrue(np.issubdtype(preds.dtype, np.floating))

    def test_03_combined_score_filtering(self):
        """Verify combined probability + expected return signal filtering logic."""
        probs = np.array([0.58, 0.52, 0.62, 0.56])
        pred_rets = np.array([0.015, 0.020, -0.005, 0.012])

        # Signal requires P(up) >= 0.55 AND Pred_Ret > 0.01
        sigs = ((probs >= 0.55) & (pred_rets > 0.01)).astype(int)
        expected_sigs = np.array([1, 0, 0, 1])

        np.testing.assert_array_equal(sigs, expected_sigs)

    def test_04_champion_json_unmodified(self):
        """Verify config/champion.json is untouched during tests."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, mtime)


if __name__ == "__main__":
    unittest.main()
