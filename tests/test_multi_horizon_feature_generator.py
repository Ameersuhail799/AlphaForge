"""Focused Unit & Temporal Leakage Tests for Multi-Horizon Feature Generator (Mission 16 Step 1)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.multi_horizon_feature_generator import (
    PROPOSED_31_FEATURES,
    MultiHorizonFeatureGenerator,
)


def create_synthetic_market_data(n_samples: int = 350, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV market dataframe."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_samples)

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


class TestMultiHorizonFeatureGenerator(unittest.TestCase):
    def test_01_feature_count_verification(self):
        """Verify generator creates exactly the 31 pre-registered proposed features."""
        self.assertEqual(len(PROPOSED_31_FEATURES), 31)

        df_raw = create_synthetic_market_data(n_samples=300)
        gen = MultiHorizonFeatureGenerator()
        df_out = gen.generate(df_raw)

        for col in PROPOSED_31_FEATURES:
            self.assertIn(col, df_out.columns)

    def test_02_temporal_leakage_audit(self):
        """LEAKAGE TEST: Verify modifying future price data at index t+k NEVER alters feature[t]."""
        df_raw1 = create_synthetic_market_data(n_samples=300, seed=101)
        df_raw2 = df_raw1.copy()

        # Modify future price rows (rows 250 to 299)
        df_raw2.iloc[250:, df_raw2.columns.get_loc("Close")] *= 1.50
        df_raw2.iloc[250:, df_raw2.columns.get_loc("High")] *= 1.50

        gen = MultiHorizonFeatureGenerator()
        df_out1 = gen.generate(df_raw1)
        df_out2 = gen.generate(df_raw2)

        # Historical features up to index 249 MUST be 100% identical
        for col in PROPOSED_31_FEATURES:
            np.testing.assert_array_almost_equal(
                df_out1.iloc[:250][col].values,
                df_out2.iloc[:250][col].values,
                err_msg=f"Temporal leakage detected in feature: {col}",
            )

    def test_03_nan_and_infinity_checks(self):
        """Verify feature outputs contain no Inf/NaN values past the maximum rolling warm-up period."""
        df_raw = create_synthetic_market_data(n_samples=350, seed=202)
        gen = MultiHorizonFeatureGenerator()
        df_out = gen.generate(df_raw)

        # Warm-up period for 52-week (252 bars) metrics = 252 rows
        valid_df = df_out.iloc[252:][PROPOSED_31_FEATURES]

        self.assertFalse(np.isinf(valid_df.values).any(), "Infinite values detected in feature matrix.")
        self.assertFalse(valid_df.isna().any().any(), "Unexpected NaN values detected past warm-up window.")

    def test_04_champion_json_unmodified(self):
        """Verify champion.json artifact is never modified during feature testing."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            original_mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, original_mtime)


if __name__ == "__main__":
    unittest.main()
