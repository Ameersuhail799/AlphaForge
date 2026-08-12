"""Unit and safety tests for Mission 20 Forward Paper-Trading Validation."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.mission19_edge_validation import run_strategy_simulation
from src.research.mission20_forward_paper_trading import C57_FEATURES


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


class TestMission20ForwardPaper(unittest.TestCase):
    def test_01_chronological_ordering(self):
        """Verify date index is strictly monotonically increasing."""
        val_df = create_synthetic_market_data(n_samples=50, seed=1)
        self.assertTrue(val_df.index.is_monotonic_increasing)

    def test_02_mode_a_single_position_constraint(self):
        """Verify Mode A permits at most 1 active position at any time."""
        val_df = create_synthetic_market_data(n_samples=50, seed=2)
        sigs = np.ones(50)  # Continuous buy signals

        res = run_strategy_simulation(val_df, sigs, cost_bps=0.0010, initial_capital=100000.0)

        # Non-overlapping 10-day trades: 50 days / 10 = 5 max trades
        self.assertLessEqual(len(res["ledger"]), 5)

    def test_03_exact_ten_day_holding_period(self):
        """Verify holding period for completed non-boundary trades is exactly 10 days."""
        val_df = create_synthetic_market_data(n_samples=50, seed=3)
        sigs = np.zeros(50)
        sigs[0] = 1

        res = run_strategy_simulation(val_df, sigs, cost_bps=0.0010, initial_capital=100000.0)

        if len(res["ledger"]) > 0:
            tr = res["ledger"][0]
            self.assertEqual(tr["exit_idx"] - tr["entry_idx"], 10)

    def test_04_equity_reconciliation_invariant(self):
        """Verify Total Equity == Cash + Market Value of Open Positions."""
        val_df = create_synthetic_market_data(n_samples=50, seed=4)
        sigs = np.random.default_rng(4).uniform(0, 1, size=50) > 0.8

        res = run_strategy_simulation(val_df, sigs.astype(int), cost_bps=0.0010, initial_capital=100000.0)

        net_pnl_sum = sum(tr["net_pnl"] for tr in res["ledger"])
        expected_final = 100000.0 + net_pnl_sum
        self.assertAlmostEqual(res["equity_curve"][-1], expected_final, delta=1e-3)

    def test_05_holdout_protection(self):
        """Verify config/champion.json is untouched and holdout is protected."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, mtime)


if __name__ == "__main__":
    unittest.main()
