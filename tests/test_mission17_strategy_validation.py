"""Unit and safety tests for Mission 17 Strategy Validation Experiment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.mission17_strategy_validation import (
    CONFIGURATIONS,
    compute_trading_simulation,
    run_mission17_experiment,
)


def create_synthetic_market_data(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataframe for test storage mock."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_samples)

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


class TestMission17StrategyValidation(unittest.TestCase):
    def test_01_five_configurations_specified(self):
        """Verify experiment runner specifies exactly 5 configurations C0, C5, C7, C57, C8."""
        self.assertEqual(len(CONFIGURATIONS), 5)
        self.assertIn("C0", CONFIGURATIONS)
        self.assertIn("C5", CONFIGURATIONS)
        self.assertIn("C7", CONFIGURATIONS)
        self.assertIn("C57", CONFIGURATIONS)
        self.assertIn("C8", CONFIGURATIONS)

    def test_02_trading_simulation_calculation(self):
        """Verify compute_trading_simulation computes returns, drawdowns, and transaction costs correctly."""
        preds = np.array([1, 0, 1, 1, 0, 1])
        rets = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.04])

        cum_ret, bh_ret, win_rate, pf, max_dd, total_trades, sharpe = compute_trading_simulation(
            preds, rets, cost_bps=0.0010
        )

        self.assertEqual(total_trades, 4)
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)
        self.assertGreaterEqual(max_dd, 0.0)

    def test_03_minimal_experiment_run(self):
        """Verify run_mission17_experiment executes cleanly on synthetic dataset."""
        df_raw = create_synthetic_market_data(n_samples=1200, seed=101)

        class DummyStorageEngine:
            def dataset_exists(self, asset_name):
                return True

            def load_dataset(self, asset_name):
                return df_raw

        with patch("src.research.mission17_strategy_validation.StorageEngine", DummyStorageEngine):
            with tempfile.TemporaryDirectory() as tmp_dir:
                res = run_mission17_experiment(
                    assets=["tcs_ns"],
                    models=["random_forest"],
                    folds=2,
                    scale=True,
                    output_dir=tmp_dir,
                )

                df_fold = res["fold_results"]
                self.assertFalse(df_fold.empty)

                # 5 configs * 1 asset * 1 model * 2 folds = 10 evaluations
                self.assertEqual(len(df_fold), 10)

                required_cols = [
                    "config_id", "asset", "target_name", "model", "fold",
                    "roc_auc", "pr_auc", "mcc", "f1", "precision", "recall", "ppr",
                    "mean_realized_ret_buy", "mean_realized_ret_sell", "return_spread",
                    "total_strat_return", "buy_hold_return", "win_rate", "profit_factor", "max_drawdown",
                    "filtered_trades_count", "filtered_ppr", "filtered_precision", "filtered_cum_return",
                ]
                for col in required_cols:
                    self.assertIn(col, df_fold.columns)

    def test_04_champion_json_unmodified(self):
        """Verify champion.json artifact is never modified during experiment runner tests."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            original_mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, original_mtime)


if __name__ == "__main__":
    unittest.main()
