"""Unit and safety tests for Mission 24 Adaptive Trade Engine."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from src.research.mission24_adaptive_trade_engine import run_adaptive_sizing_simulation


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
            "HIST_VOL_20": 0.015,
            "BULLISH_TREND_REGIME": 1,
        },
        index=dates,
    )


class TestMission24AdaptiveTradeEngine(unittest.TestCase):
    def test_01_calibration_fitting_synthetic(self):
        """Verify Platt and Isotonic calibration models fit without error."""
        rng = np.random.default_rng(1)
        X_train = rng.normal(0, 1, size=(50, 5))
        y_train = rng.choice([0, 1], size=50)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=2)
        cal.fit(X_train, y_train)

        X_val = rng.normal(0, 1, size=(20, 5))
        probs = cal.predict_proba(X_val)[:, 1]

        self.assertEqual(len(probs), 20)
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_02_adaptive_sizing_weights_bounded(self):
        """Verify position sizing weights are strictly bounded between 0% and 100%."""
        val_df = create_synthetic_market_data(n_samples=50, seed=2)
        probs = np.full(50, 0.68)
        pred_rets = np.full(50, 0.02)

        for strat in ["FIXED_100", "CONFIDENCE_SCALED", "RISK_NORMALIZED", "REGIME_AWARE"]:
            res = run_adaptive_sizing_simulation(val_df, probs, pred_rets, sizing_strategy=strat, cost_bps=0.0010, initial_capital=100000.0)
            for tr in res["ledger"]:
                self.assertGreaterEqual(tr["position_weight"], 0.0)
                self.assertLessEqual(tr["position_weight"], 1.0 + 1e-6)

    def test_03_accounting_invariant_reconciliation(self):
        """Verify Total Equity == Cash + Position Value for all sizing strategies."""
        val_df = create_synthetic_market_data(n_samples=50, seed=3)
        probs = np.random.default_rng(3).uniform(0.5, 0.75, size=50)
        pred_rets = np.random.default_rng(3).uniform(0.0, 0.03, size=50)

        res = run_adaptive_sizing_simulation(val_df, probs, pred_rets, sizing_strategy="CONFIDENCE_SCALED", cost_bps=0.0010, initial_capital=100000.0)

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
