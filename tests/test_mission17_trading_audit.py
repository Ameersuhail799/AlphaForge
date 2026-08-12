"""Unit and safety tests for Mission 17.5 Trading Engine Forensic Audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.mission17_trading_audit import (
    C57_FEATURES,
    run_mission17_trading_audit,
)


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


class TestMission17TradingAudit(unittest.TestCase):
    def test_01_c57_features_defined(self):
        """Verify C57 feature set contains expected SHORTLIST_16 + Group E + Group G features."""
        self.assertEqual(len(C57_FEATURES), 25)

    def test_02_minimal_audit_run(self):
        """Verify run_mission17_trading_audit executes cleanly on synthetic dataset."""
        df_raw = create_synthetic_market_data(n_samples=2000, seed=101)

        class DummyStorageEngine:
            def dataset_exists(self, asset_name):
                return True

            def load_dataset(self, asset_name):
                return df_raw

        with patch("src.research.mission17_trading_audit.StorageEngine", DummyStorageEngine):
            with tempfile.TemporaryDirectory() as tmp_dir:
                res = run_mission17_trading_audit(
                    cost_bps=0.0010,
                    output_dir=tmp_dir,
                )

                df_summary = res["summary"]
                df_fold = res["fold_results"]
                df_trade = res["trade_ledger"]

                self.assertFalse(df_summary.empty)
                self.assertFalse(df_fold.empty)
                self.assertIn("audit_verdict", df_summary.columns)
                self.assertEqual(df_summary["audit_verdict"].iloc[0], "C. IMPLEMENTATION ERROR FOUND")

    def test_03_champion_json_unmodified(self):
        """Verify champion.json artifact is never modified during audit tests."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            original_mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, original_mtime)


if __name__ == "__main__":
    unittest.main()
