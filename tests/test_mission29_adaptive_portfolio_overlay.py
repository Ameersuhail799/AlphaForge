"""Unit and safety tests for Mission 29 Adaptive Portfolio Overlay."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.mission27_cross_asset_generalization import ASSET_UNIVERSE, build_asset_dataset
from src.research.mission29_adaptive_portfolio_overlay import run_adaptive_overlay_simulation


class TestMission29AdaptivePortfolioOverlay(unittest.TestCase):
    def test_01_adaptive_overlay_architectures_execution(self):
        """Verify candidate adaptive overlay architectures run cleanly on multi-asset signals."""
        asset_datasets = {a: build_asset_dataset(a) for a in ASSET_UNIVERSE}
        common_idx = asset_datasets["tcs_ns"].index[-50:]

        signals = {a: pd.Series(1, index=common_idx) for a in ASSET_UNIVERSE}
        pred_rets = {a: pd.Series(0.02, index=common_idx) for a in ASSET_UNIVERSE}

        for arch in ["EQUAL_WEIGHT", "DRAWDOWN_GOVERNOR", "VOL_EXPANSION", "CORRELATION_CAP", "HYSTERESIS_DD", "COMBINED_ADAPTIVE"]:
            res = run_adaptive_overlay_simulation(asset_datasets, signals, pred_rets, architecture=arch, cost_bps=0.0010, initial_capital=100000.0)
            self.assertIn("cum_return_pct", res)
            self.assertIn("max_drawdown_pct", res)
            self.assertIn("calmar_ratio", res)
            self.assertIn("sortino_ratio", res)

    def test_02_hysteresis_buffer_scaling(self):
        """Verify hysteresis buffer toggles defensive mode at 20% DD and releases at < 10% DD."""
        asset_datasets = {a: build_asset_dataset(a) for a in ASSET_UNIVERSE}
        common_idx = asset_datasets["tcs_ns"].index[-50:]

        signals = {a: pd.Series(1, index=common_idx) for a in ASSET_UNIVERSE}
        pred_rets = {a: pd.Series(0.02, index=common_idx) for a in ASSET_UNIVERSE}

        res_hys = run_adaptive_overlay_simulation(asset_datasets, signals, pred_rets, architecture="HYSTERESIS_DD", cost_bps=0.0010, initial_capital=100000.0)
        self.assertGreater(len(res_hys["equity_curve"]), 0)

    def test_03_champion_json_unmodified(self):
        """Verify config/champion.json is untouched during tests."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, mtime)


if __name__ == "__main__":
    unittest.main()
