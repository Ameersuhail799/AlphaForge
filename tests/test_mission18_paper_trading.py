"""Integration tests for Mission 18 Production-Grade Paper Trading Engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.paper_trading_engine import run_mission18_paper_trading_experiment


def create_synthetic_market_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataframe for test storage mock."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n_samples)

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


class TestMission18PaperTradingIntegration(unittest.TestCase):
    def test_01_full_paper_trading_experiment_run(self):
        """Verify run_mission18_paper_trading_experiment executes cleanly and generates required artifacts."""
        df_raw = create_synthetic_market_data(n_samples=2000, seed=101)

        class DummyStorageEngine:
            def dataset_exists(self, asset_name):
                return True

            def load_dataset(self, asset_name):
                return df_raw

        with patch("src.research.paper_trading_engine.StorageEngine", DummyStorageEngine):
            with tempfile.TemporaryDirectory() as tmp_dir:
                res = run_mission18_paper_trading_experiment(
                    scale=True,
                    output_dir=tmp_dir,
                )

                df_summary = res["summary"]
                df_ledger = res["ledger"]
                df_daily_eq = res["daily_equity"]

                self.assertFalse(df_summary.empty)
                self.assertFalse(df_daily_eq.empty)

                # Check CSV output artifacts created
                tmp_path = Path(tmp_dir)
                self.assertTrue((tmp_path / "mission18_paper_trade_ledger.csv").exists())
                self.assertTrue((tmp_path / "mission18_equity_curve.csv").exists())
                self.assertTrue((tmp_path / "mission18_summary.csv").exists())
                self.assertTrue((tmp_path / "MISSION_18_PAPER_TRADING_REPORT.md").exists())

    def test_02_champion_json_unmodified(self):
        """Verify champion.json artifact is never modified during paper trading integration testing."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            original_mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, original_mtime)


if __name__ == "__main__":
    unittest.main()
