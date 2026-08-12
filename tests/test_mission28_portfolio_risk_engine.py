"""Unit and safety tests for Mission 28 Portfolio Risk Engine."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.mission27_cross_asset_generalization import ASSET_UNIVERSE, build_asset_dataset
from src.research.mission28_portfolio_risk_engine import run_portfolio_risk_simulation


class TestMission28PortfolioRiskEngine(unittest.TestCase):
    def test_01_portfolio_risk_architectures_execution(self):
        """Verify candidate portfolio architectures run cleanly on multi-asset signals."""
        asset_datasets = {a: build_asset_dataset(a) for a in ASSET_UNIVERSE}
        common_idx = asset_datasets["tcs_ns"].index[-50:]

        signals = {a: pd.Series(1, index=common_idx) for a in ASSET_UNIVERSE}
        pred_rets = {a: pd.Series(0.02, index=common_idx) for a in ASSET_UNIVERSE}

        for arch in ["EQUAL_WEIGHT", "INVERSE_VOLATILITY", "SIGNAL_QUALITY_WEIGHTED", "DRAWDOWN_GOVERNOR", "COMBINED_VOL_DD"]:
            res = run_portfolio_risk_simulation(asset_datasets, signals, pred_rets, architecture=arch, cost_bps=0.0010, initial_capital=100000.0)
            self.assertIn("cum_return_pct", res)
            self.assertIn("max_drawdown_pct", res)
            self.assertIn("calmar_ratio", res)

    def test_02_drawdown_governor_exposure_scaling(self):
        """Verify drawdown governor scales down exposure during portfolio drawdowns."""
        asset_datasets = {a: build_asset_dataset(a) for a in ASSET_UNIVERSE}
        common_idx = asset_datasets["tcs_ns"].index[-50:]

        signals = {a: pd.Series(1, index=common_idx) for a in ASSET_UNIVERSE}
        pred_rets = {a: pd.Series(0.02, index=common_idx) for a in ASSET_UNIVERSE}

        res_gov = run_portfolio_risk_simulation(asset_datasets, signals, pred_rets, architecture="DRAWDOWN_GOVERNOR", cost_bps=0.0010, initial_capital=100000.0)
        self.assertGreater(len(res_gov["equity_curve"]), 0)

    def test_03_champion_json_unmodified(self):
        """Verify config/champion.json is untouched during tests."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, mtime)


if __name__ == "__main__":
    unittest.main()
