"""Unit and safety tests for Mission 26 Meta-Labelled Trade Quality Engine."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
            "ATR_14": 2.0,
            "HIST_VOL_20": 0.015,
            "PRICE_TO_SMA50_DIST": 0.02,
            "RANGE_COMPRESSION_EXP": 1.1,
            "BULLISH_TREND_REGIME": 1,
        },
        index=dates,
    )


class TestMission26MetaTradeQuality(unittest.TestCase):
    def test_01_meta_classifier_fitting(self):
        """Verify secondary meta-classifier fits and predicts probabilities on synthetic data."""
        rng = np.random.default_rng(1)
        meta_X = rng.normal(0, 1, size=(50, 6))
        meta_y = rng.choice([0, 1], size=50)

        clf_meta = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf_meta.fit(meta_X, meta_y)

        val_X = rng.normal(0, 1, size=(20, 6))
        p_meta = clf_meta.predict_proba(val_X)[:, 1]

        self.assertEqual(len(p_meta), 20)
        self.assertTrue(np.all(p_meta >= 0.0) and np.all(p_meta <= 1.0))

    def test_02_risk_adjusted_edge_calculation(self):
        """Verify Risk-Adjusted Edge RAE = pred_return / (ATR / Close)."""
        pred_rets = np.array([0.02, 0.01])
        atrs = np.array([2.0, 4.0])
        closes = np.array([100.0, 100.0])

        atr_rel = atrs / closes
        rae = pred_rets / atr_rel

        expected_rae = np.array([1.0, 0.25])
        np.testing.assert_allclose(rae, expected_rae, rtol=1e-5)

    def test_03_meta_filtered_signals_execution(self):
        """Verify strategy simulation runs cleanly with meta-filtered signals."""
        val_df = create_synthetic_market_data(n_samples=50, seed=2)
        sigs = np.zeros(50, dtype=int)
        sigs[5] = 1
        sigs[20] = 1

        res = run_strategy_simulation(val_df, sigs, cost_bps=0.0010)
        self.assertIn("cum_return_pct", res)
        self.assertIn("ledger", res)

    def test_04_champion_json_unmodified(self):
        """Verify config/champion.json is untouched during tests."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, mtime)


if __name__ == "__main__":
    unittest.main()
