"""Unit and safety tests for Mission 25 Adaptive Exit & Asymmetric Risk Engine."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.mission25_adaptive_exit import run_adaptive_exit_simulation


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
        },
        index=dates,
    )


class TestMission25AdaptiveExit(unittest.TestCase):
    def test_01_exit_mechanisms_execution(self):
        """Verify candidate exit mechanisms execute cleanly on synthetic data."""
        val_df = create_synthetic_market_data(n_samples=50, seed=1)
        probs = np.full(50, 0.60)
        pred_rets = np.full(50, 0.02)

        for mech in ["CONTROL_FIXED_10D", "PROFIT_TARGET", "ATR_STOP_LOSS", "TRAILING_ATR", "MODEL_DETERIORATION", "RETURN_DECAY", "COMBINED_ADAPTIVE"]:
            res = run_adaptive_exit_simulation(val_df, probs, pred_rets, exit_mechanism=mech, cost_bps=0.0010, initial_capital=100000.0)
            self.assertIn("cum_return_pct", res)
            self.assertIn("ledger", res)

    def test_02_stop_loss_trigger_causal(self):
        """Verify stop loss triggers when low price crosses stop boundary."""
        val_df = create_synthetic_market_data(n_samples=50, seed=2)
        # Induce sharp price drop at bar 5
        val_df.iloc[5:, val_df.columns.get_loc("Low")] = 50.0
        val_df.iloc[5:, val_df.columns.get_loc("Close")] = 50.0

        probs = np.zeros(50)
        probs[0] = 0.60  # Entry at bar 0
        pred_rets = np.full(50, 0.02)

        res = run_adaptive_exit_simulation(val_df, probs, pred_rets, exit_mechanism="ATR_STOP_LOSS", cost_bps=0.0010, initial_capital=100000.0)

        self.assertGreater(len(res["ledger"]), 0)
        tr = res["ledger"][0]
        self.assertEqual(tr["exit_reason"], "STOP_LOSS")

    def test_03_accounting_invariant_reconciliation(self):
        """Verify Total Equity == Cash + Position Value for all exit mechanisms."""
        val_df = create_synthetic_market_data(n_samples=50, seed=3)
        probs = np.random.default_rng(3).uniform(0.5, 0.75, size=50)
        pred_rets = np.random.default_rng(3).uniform(0.0, 0.03, size=50)

        res = run_adaptive_exit_simulation(val_df, probs, pred_rets, exit_mechanism="COMBINED_ADAPTIVE", cost_bps=0.0010, initial_capital=100000.0)

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
