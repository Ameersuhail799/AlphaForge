"""Unit Tests for Strategy Tester Module.

Verifies:
1. Template signal generation across all 4 strategy templates.
2. Integration with exact 2026 NSE delivery cost calculations.
3. Side-by-side comparison metrics against Buy-and-Hold benchmark.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from src.research.strategy_tester import (
    compute_rsi,
    generate_template_signals,
    run_strategy_backtest,
)
from src.research.benchmark_and_cost_reality_check import calculate_nse_delivery_cost


class TestStrategyTester(unittest.TestCase):
    """Test suite for Strategy Tester module."""

    def test_compute_rsi(self) -> None:
        """Test RSI calculation returns values bounded between 0 and 100."""
        prices = pd.Series([100 + i + (i % 3) for i in range(30)])
        rsi = compute_rsi(prices, period=14)
        self.assertEqual(len(rsi), 30)
        self.assertTrue(all(0.0 <= val <= 100.0 for val in rsi))

    def test_template_signals_rsi_threshold(self) -> None:
        """Test rsi_threshold template signal generation."""
        df = pd.DataFrame({
            "Close": [100.0 - i * 2 for i in range(20)],
            "Open": [100.0 - i * 2 + 1 for i in range(20)],
            "Volume": [1000] * 20,
        })
        sigs = generate_template_signals(df, "rsi_threshold", {"period": 5, "buy_threshold": 30})
        self.assertEqual(len(sigs), 20)
        self.assertIn(1, sigs.values)

    def test_template_signals_sma_crossover(self) -> None:
        """Test sma_crossover template signal generation."""
        # Downtrend then sharp uptrend to produce crossover
        closes = [100.0 - i for i in range(30)] + [70.0 + i * 5 for i in range(30)]
        df = pd.DataFrame({
            "Close": closes,
            "Open": closes,
            "Volume": [1000] * 60,
        })
        sigs = generate_template_signals(df, "sma_crossover", {"fast": 5, "slow": 15})
        self.assertEqual(len(sigs), 60)
        self.assertIn(1, sigs.values)

    def test_template_signals_price_vs_sma(self) -> None:
        """Test price_vs_sma template signal generation."""
        closes = [100.0] * 20 + [150.0] * 10
        df = pd.DataFrame({
            "Close": closes,
            "Open": closes,
            "Volume": [1000] * 30,
        })
        sigs = generate_template_signals(df, "price_vs_sma", {"period": 10})
        self.assertEqual(len(sigs), 30)
        self.assertIn(1, sigs.values)

    def test_template_signals_volume_breakout(self) -> None:
        """Test volume_breakout template signal generation."""
        vols = [1000] * 25 + [5000] + [1000] * 4
        closes = [100.0] * 25 + [105.0] + [105.0] * 4
        opens = [100.0] * 25 + [101.0] + [105.0] * 4
        df = pd.DataFrame({
            "Close": closes,
            "Open": opens,
            "Volume": vols,
        })
        sigs = generate_template_signals(df, "volume_breakout", {"multiplier": 2.0})
        self.assertEqual(len(sigs), 30)
        self.assertEqual(sigs.iloc[25], 1)

    def test_nse_cost_function_integration(self) -> None:
        """Verify that calculate_nse_delivery_cost applies non-zero entry and exit fees."""
        trade_val = 20000.0
        entry_cost = calculate_nse_delivery_cost(trade_val, is_buy=True)
        exit_cost = calculate_nse_delivery_cost(trade_val, is_buy=False)

        # STT 0.1% + Stamp 0.015% + Exch/SEBI ~ 0.11864% -> ~23.73 INR
        self.assertGreater(entry_cost, 20.0)
        # STT 0.1% + Exch/SEBI + Flat DP INR 15.93 -> ~36.64 INR
        self.assertGreater(exit_cost, 30.0)

    def test_run_strategy_backtest_single_asset(self) -> None:
        """Test end-to-end strategy backtest on single asset."""
        res = run_strategy_backtest(
            asset_selection="tcs_ns",
            template="rsi_threshold",
            params={"period": 14, "buy_threshold": 30},
            exit_days=10,
        )

        self.assertEqual(res["asset_selection"], "tcs_ns")
        self.assertIn("verdict", res)
        self.assertIn("strategy_metrics", res)
        self.assertIn("benchmark_metrics", res)

        strat_m = res["strategy_metrics"]
        bh_m = res["benchmark_metrics"]

        # Check required fields
        for field in ["cagr_pct", "sharpe", "sortino", "max_dd_pct", "total_trades", "win_rate_pct"]:
            self.assertIn(field, strat_m)
            self.assertIn(field, bh_m)

    def test_run_strategy_backtest_pooled_assets(self) -> None:
        """Test end-to-end strategy backtest pooled across all 5 assets."""
        res = run_strategy_backtest(
            asset_selection="all_pooled",
            template="sma_crossover",
            params={"fast": 20, "slow": 50},
            exit_days=14,
        )

        self.assertEqual(res["asset_selection"], "all_pooled")
        self.assertEqual(res["asset_display_name"], "All 5 Equities Pooled")
        self.assertTrue(res["evaluation_period"]["trading_days"] > 5000)


if __name__ == "__main__":
    unittest.main()
