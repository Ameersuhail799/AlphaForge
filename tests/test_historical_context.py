"""Unit tests for Historical Context Research Module."""

from __future__ import annotations

import unittest
from src.research.historical_context import compute_historical_context


class TestHistoricalContext(unittest.TestCase):

    def test_tcs_historical_context(self):
        res = compute_historical_context("tcs_ns")
        self.assertEqual(res["symbol"], "tcs_ns")
        self.assertEqual(res["display_name"], "TCS")
        self.assertIn("regime_description", res)
        self.assertIn("sample_count", res)
        self.assertIsInstance(res["insufficient_sample"], bool)

        if not res["insufficient_sample"]:
            self.assertGreaterEqual(res["sample_count"], 20)
            metrics = res["metrics"]
            self.assertIsNotNone(metrics)
            self.assertIn("pct_positive", metrics)
            self.assertIn("min_ret_pct", metrics)
            self.assertIn("p25_ret_pct", metrics)
            self.assertIn("median_ret_pct", metrics)
            self.assertIn("p75_ret_pct", metrics)
            self.assertIn("max_ret_pct", metrics)

    def test_infy_historical_context(self):
        res = compute_historical_context("infy_ns")
        self.assertEqual(res["symbol"], "infy_ns")
        self.assertEqual(res["display_name"], "INFY")
        self.assertIn("regime_description", res)
        self.assertIn("sample_count", res)
        self.assertIsInstance(res["insufficient_sample"], bool)

    def test_invalid_symbol_fallback(self):
        res = compute_historical_context("invalid_symbol")
        self.assertEqual(res["symbol"], "tcs_ns")


if __name__ == "__main__":
    unittest.main()
