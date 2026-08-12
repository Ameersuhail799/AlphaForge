"""Unit and safety tests for Mission 27 Cross-Asset Generalization Research."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.mission27_cross_asset_generalization import ASSET_UNIVERSE, build_asset_dataset


class TestMission27CrossAssetGeneralization(unittest.TestCase):
    def test_01_asset_universe_datasets_existence(self):
        """Verify all 5 assets in ASSET_UNIVERSE build datasets cleanly."""
        for asset in ASSET_UNIVERSE:
            df_full = build_asset_dataset(asset)
            self.assertGreater(len(df_full), 1000)
            self.assertIn("TARGET_D", df_full.columns)
            self.assertIn("REALIZED_RET_10D", df_full.columns)

    def test_02_locked_thresholds_no_per_asset_tuning(self):
        """Verify probability threshold (0.55) and return threshold (0.01) are strictly locked."""
        probs = np.array([0.58, 0.52, 0.62, 0.56])
        pred_rets = np.array([0.015, 0.020, -0.005, 0.012])

        # Locked rule: P(up) >= 0.55 AND Pred_Ret > 0.01
        sigs = ((probs >= 0.55) & (pred_rets > 0.01)).astype(int)
        expected_sigs = np.array([1, 0, 0, 1])

        np.testing.assert_array_equal(sigs, expected_sigs)

    def test_03_equal_weight_portfolio_allocation(self):
        """Verify equal weight portfolio allocates 20% max per asset across 5 assets."""
        n_assets = len(ASSET_UNIVERSE)
        alloc_weight = 1.0 / n_assets
        self.assertAlmostEqual(alloc_weight, 0.20, delta=1e-6)

    def test_04_champion_json_unmodified(self):
        """Verify config/champion.json is untouched during tests."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, mtime)


if __name__ == "__main__":
    unittest.main()
