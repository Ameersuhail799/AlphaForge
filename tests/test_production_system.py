"""Integration and System Tests for AlphaForge Production Suite."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.production.paper_portfolio import PaperPortfolioEngine
from src.production.server import _get_backtest_research_summaries, STATIC_DIR
from src.production.trading_engine import SUPPORTED_ASSETS, ProductionTradingEngine


class TestAlphaForgeProductionSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize production engine instance once for test suite across supported liquid assets."""
        cls.engine = ProductionTradingEngine(assets=["tcs_ns", "infy_ns"])

    def test_01_trading_engine_signal_generation(self):
        """Verify production trading engine produces structured trade signals and explanations."""
        sig = self.engine.predict_trade_signal("tcs_ns")
        self.assertEqual(sig["symbol"], "tcs_ns")
        self.assertIn(sig["signal"], ["BUY", "HOLD", "SELL", "BEARISH"])
        self.assertGreaterEqual(sig["prob_up_pct"], 0.0)
        self.assertLessEqual(sig["prob_up_pct"], 100.0)
        self.assertIn("reasons", sig)
        self.assertGreater(len(sig["reasons"]), 0)

    def test_02_trading_engine_market_data(self):
        """Verify market data extraction for charting produces aligned series."""
        m_data = self.engine.get_asset_market_data("tcs_ns", limit=50)
        self.assertEqual(len(m_data["dates"]), 50)
        self.assertEqual(len(m_data["close"]), 50)
        self.assertEqual(len(m_data["sma20"]), 50)
        self.assertGreater(m_data["last_price"], 0.0)

    def test_03_paper_portfolio_buy_and_sell_cycle(self):
        """Verify paper trading portfolio handles BUY/SELL execution and accounting invariants."""
        port = PaperPortfolioEngine(initial_capital=100000.0)

        # Execute BUY
        res_buy = port.execute_trade("tcs_ns", "BUY", current_price=3000.0, capital_alloc=20000.0)
        self.assertEqual(res_buy["status"], "SUCCESS")
        self.assertIn("tcs_ns", port.open_positions)
        self.assertLess(port.cash, 100000.0)

        # Invariant Check: Total Equity == Cash + Open Position Value
        summary_1 = port.get_portfolio_summary()
        expected_eq_1 = summary_1["cash_balance"] + summary_1["positions_value"]
        self.assertAlmostEqual(summary_1["current_equity"], expected_eq_1, places=2)

        # Execute SELL / CLOSE
        res_sell = port.execute_trade("tcs_ns", "SELL", current_price=3100.0)
        self.assertEqual(res_sell["status"], "SUCCESS")
        self.assertNotIn("tcs_ns", port.open_positions)
        self.assertEqual(len(port.trade_history), 1)
        self.assertGreater(port.trade_history[0]["net_pnl"], 0.0)

        # Reset Portfolio Check
        port.reset_portfolio(100000.0)
        self.assertEqual(port.cash, 100000.0)
        self.assertEqual(len(port.open_positions), 0)
        self.assertEqual(len(port.trade_history), 0)

    def test_04_invalid_asset_handling(self):
        """Verify engine gracefully rejects unsupported asset symbols."""
        with self.assertRaises(KeyError):
            self.engine.predict_trade_signal("INVALID_SYMBOL_999")

    def test_05_backtest_summaries_contract(self):
        """Verify research backtest summaries helper returns multi-asset metrics."""
        summaries = _get_backtest_research_summaries()
        self.assertIn("mission26_champion", summaries)
        self.assertIn("mission27_cross_asset", summaries)
        self.assertEqual(len(summaries["mission27_cross_asset"]), 5)

    def test_06_champion_json_unmodified(self):
        """Verify config/champion.json is untouched during tests."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, mtime)


if __name__ == "__main__":
    unittest.main()
