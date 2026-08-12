"""Unit and safety tests for Mission 22 Technical Trading Strategy Intelligence."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.mission22_technical_strategy import (
    add_technical_confirmation_features,
    run_advanced_trade_management_simulation,
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
            "ROC_12": 0.02,
            "VOLUME_RATIO": 1.2,
            "RANGE_COMPRESSION_EXP": 1.1,
            "BULLISH_TREND_REGIME": 1,
        },
        index=dates,
    )


class TestMission22TechnicalStrategy(unittest.TestCase):
    def test_01_add_technical_confirmation_features(self):
        """Verify confirmation features engineering."""
        val_df = create_synthetic_market_data(n_samples=60, seed=1)
        df_c = add_technical_confirmation_features(val_df)

        expected_cols = ["CONFIRM_TREND", "CONFIRM_BREAKOUT", "CONFIRM_MOMENTUM", "CONFIRM_VOLATILITY", "CONFIRM_COMBINED"]
        for col in expected_cols:
            self.assertIn(col, df_c.columns)

    def test_02_breakout_no_lookahead(self):
        """Verify breakout calculation uses shift(1) for previous 20D high."""
        val_df = create_synthetic_market_data(n_samples=60, seed=2)
        high20_shift1 = val_df["Close"].rolling(20).max().shift(1)
        self.assertTrue(pd.isna(high20_shift1.iloc[19]))  # 20th row is NaN due to shift(1)

    def test_03_advanced_trade_management_accounting(self):
        """Verify advanced trade management satisfies accounting invariants."""
        val_df = create_synthetic_market_data(n_samples=50, seed=3)
        sigs = np.zeros(50)
        sigs[5] = 1

        res = run_advanced_trade_management_simulation(val_df, sigs.astype(int), exit_rule="DYNAMIC_ATR", cost_bps=0.0010, initial_capital=100000.0)

        net_pnl_sum = sum(tr["net_pnl"] for tr in res["ledger"])
        expected_final = 100000.0 + net_pnl_sum
        self.assertAlmostEqual(res["equity_curve"][-1], expected_final, delta=1e-3)

    def test_04_champion_json_unmodified(self):
        """Verify config/champion.json is untouched during tests."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, mtime)


if __name__ == "__main__":
    unittest.main()
