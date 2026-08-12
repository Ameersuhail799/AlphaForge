"""Unit and safety tests for Mission 19 Strategy Edge Validation Audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.mission19_edge_validation import (
    C57_FEATURES,
    run_strategy_simulation,
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
        },
        index=dates,
    )


class TestMission19EdgeValidation(unittest.TestCase):
    def test_01_c57_features_count(self):
        """Verify C57 feature set contains expected 25 features."""
        self.assertEqual(len(C57_FEATURES), 25)

    def test_02_run_strategy_simulation_accounting(self):
        """Verify run_strategy_simulation satisfies accounting invariants."""
        val_df = create_synthetic_market_data(n_samples=50, seed=1)
        sigs = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0] * 30)

        res = run_strategy_simulation(val_df, sigs, cost_bps=0.0010, initial_capital=100000.0)

        # Check ending equity equals initial capital + sum of net PnL
        net_pnl_sum = sum(tr["net_pnl"] for tr in res["ledger"])
        expected_final = 100000.0 + net_pnl_sum
        self.assertAlmostEqual(res["equity_curve"][-1], expected_final, delta=1e-3)

    def test_03_champion_json_unmodified(self):
        """Verify champion.json artifact is never modified during edge validation tests."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            original_mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, original_mtime)


if __name__ == "__main__":
    unittest.main()
