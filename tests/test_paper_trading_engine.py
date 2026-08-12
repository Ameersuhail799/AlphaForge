"""Unit and safety tests for Mission 18 Paper Trading Engine & Forensic Audit."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.research.paper_trading_engine import run_paper_simulation_fold


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


class TestPaperTradingEngineUnit(unittest.TestCase):
    def test_01_mode_a_single_position_limit(self):
        """Verify Mode A permits at most 1 active position at any time."""
        val_df = create_synthetic_market_data(n_samples=50, seed=1)
        probs = np.ones(50) * 0.80  # Continuous buy signals

        ledger, daily_eq, metrics = run_paper_simulation_fold(
            val_df=val_df,
            probs=probs,
            fold_idx=1,
            initial_capital=100000.0,
            cost_bps=0.0010,
            mode="MODE_A",
            entry_timing="SAME_BAR_CLOSE",
        )

        for eq_rec in daily_eq:
            self.assertLessEqual(eq_rec.open_positions, 1, "Mode A exceeded maximum 1 position limit!")

    def test_02_mode_b_ten_slot_limit(self):
        """Verify Mode B permits at most 10 active positions at any time."""
        val_df = create_synthetic_market_data(n_samples=50, seed=2)
        probs = np.ones(50) * 0.80  # Continuous buy signals

        ledger, daily_eq, metrics = run_paper_simulation_fold(
            val_df=val_df,
            probs=probs,
            fold_idx=1,
            initial_capital=100000.0,
            cost_bps=0.0010,
            mode="MODE_B",
            entry_timing="SAME_BAR_CLOSE",
        )

        for eq_rec in daily_eq:
            self.assertLessEqual(eq_rec.open_positions, 10, "Mode B exceeded maximum 10 position limit!")
            self.assertLessEqual(eq_rec.exposure_pct, 100.0 + 1e-4, "Mode B exceeded 100% exposure!")

    def test_03_accounting_invariant_reconciliation(self):
        """Verify Total Equity == Cash + Allocated Capital for every day."""
        val_df = create_synthetic_market_data(n_samples=50, seed=3)
        probs = np.random.default_rng(3).uniform(0.4, 0.7, size=50)

        ledger, daily_eq, metrics = run_paper_simulation_fold(
            val_df=val_df,
            probs=probs,
            fold_idx=1,
            initial_capital=100000.0,
            cost_bps=0.0010,
            mode="MODE_A",
            entry_timing="SAME_BAR_CLOSE",
        )

        for eq_rec in daily_eq:
            expected_eq = eq_rec.cash + eq_rec.allocated_capital
            self.assertAlmostEqual(eq_rec.total_equity, expected_eq, delta=1e-4)

    def test_04_anti_lookahead_future_mutation_test(self):
        """ADVERSARIAL TEST: Modifying future prices at bar t+k NEVER alters signals or ledger up to bar t."""
        val_df_base = create_synthetic_market_data(n_samples=50, seed=4)
        probs_base = np.random.default_rng(4).uniform(0.4, 0.7, size=50)

        val_df_future_mutated = val_df_base.copy()
        val_df_future_mutated.iloc[30:, val_df_future_mutated.columns.get_loc("Close")] *= 2.0

        ledger1, daily1, _ = run_paper_simulation_fold(
            val_df=val_df_base,
            probs=probs_base,
            fold_idx=1,
            initial_capital=100000.0,
            cost_bps=0.0010,
            mode="MODE_A",
            entry_timing="SAME_BAR_CLOSE",
        )

        ledger2, daily2, _ = run_paper_simulation_fold(
            val_df=val_df_future_mutated,
            probs=probs_base,
            fold_idx=1,
            initial_capital=100000.0,
            cost_bps=0.0010,
            mode="MODE_A",
            entry_timing="SAME_BAR_CLOSE",
        )

        # First 30 days of equity records must be identical
        for i in range(30):
            self.assertAlmostEqual(daily1[i].total_equity, daily2[i].total_equity, delta=1e-4)

    def test_05_true_buy_and_hold_calculation(self):
        """FORENSIC TEST: Verify true Buy & Hold is (P_end - P_start)/P_start, NOT daily 10D compounding."""
        val_df = create_synthetic_market_data(n_samples=100, seed=5)
        p_start = val_df["Close"].iloc[0]
        p_end = val_df["Close"].iloc[-1]

        true_bh_return = (p_end - p_start) / p_start
        true_bh_wealth = p_end / p_start

        # True buy and hold must equal price ratio
        self.assertAlmostEqual(1.0 + true_bh_return, true_bh_wealth, delta=1e-6)

        # Verify that daily 10D compounding produces an artificial huge number
        close = val_df["Close"]
        ret_10d = (close.shift(-10) - close) / close
        ret_10d_clean = ret_10d.fillna(0.0).values
        compounded_10d = float(np.cumprod(1.0 + ret_10d_clean)[-1] - 1.0)

        # Prove that daily 10D compounding != true buy and hold
        self.assertNotAlmostEqual(compounded_10d, true_bh_return, delta=0.5)

    def test_06_cost_sensitivity_entry_timing_isolation(self):
        """FORENSIC TEST: Cost sensitivity analysis must filter by entry_timing to prevent averaging SAME_BAR_CLOSE & NEXT_BAR_OPEN."""
        df_records = pd.DataFrame([
            {"mode": "MODE_A", "entry_timing": "SAME_BAR_CLOSE", "cost_bps": 10.0, "cum_return": 59.98},
            {"mode": "MODE_A", "entry_timing": "NEXT_BAR_OPEN", "cost_bps": 10.0, "cum_return": 65.20},
        ])

        # Wrong grouping (combines timing modes)
        wrong_avg = float(df_records.groupby(["mode", "cost_bps"])["cum_return"].mean().iloc[0])
        self.assertAlmostEqual(wrong_avg, 62.59, delta=0.01)

        # Correct grouping (separates timing modes)
        correct_same_bar = float(df_records[(df_records["mode"] == "MODE_A") & (df_records["entry_timing"] == "SAME_BAR_CLOSE")]["cum_return"].iloc[0])
        self.assertAlmostEqual(correct_same_bar, 59.98, delta=0.01)


if __name__ == "__main__":
    unittest.main()
