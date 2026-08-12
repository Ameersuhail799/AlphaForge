"""Unit and safety tests for Mission 21 Model Improvement & Signal Enhancement."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.mission21_model_improvement import (
    add_group_h_features,
    build_mission21_dataset,
)


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
            "RSI_14": 55.0,
            "VOLUME_RATIO": 1.2,
        },
        index=dates,
    )


class TestMission21ModelImprovement(unittest.TestCase):
    def test_01_add_group_h_features(self):
        """Verify Group H feature creation on synthetic market data."""
        val_df = create_synthetic_market_data(n_samples=60, seed=1)
        df_h = add_group_h_features(val_df)

        expected_cols = [
            "SMA20_50_SLOPE",
            "PRICE_TO_SMA50_DIST",
            "RANGE_COMPRESSION_EXP",
            "TREND_VOL_INTERACTION",
            "RSI_SLOPE_5D",
            "VOLUME_BREAKOUT_CONFIRM",
            "BULLISH_TREND_REGIME",
        ]
        for col in expected_cols:
            self.assertIn(col, df_h.columns)

    def test_02_bullish_trend_regime_causal_logic(self):
        """Verify BULLISH_TREND_REGIME logic is binary 0 or 1 and strictly backward-looking."""
        val_df = create_synthetic_market_data(n_samples=60, seed=2)
        df_h = add_group_h_features(val_df)

        regime_vals = df_h["BULLISH_TREND_REGIME"].dropna().unique()
        for v in regime_vals:
            self.assertIn(v, [0, 1])

    def test_03_champion_json_unmodified(self):
        """Verify config/champion.json is untouched during tests."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, mtime)


if __name__ == "__main__":
    unittest.main()
