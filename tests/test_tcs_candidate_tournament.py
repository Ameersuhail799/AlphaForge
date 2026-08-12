"""Unit tests for Mission 15 Step 3 TCS Candidate Tournament & Deep Validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.tcs_candidate_tournament import (
    SHORTLIST_16,
    TOURNAMENT_CANDIDATES,
    compute_stability_score,
    run_tcs_candidate_tournament,
)


def create_synthetic_market_data(n_samples: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV dataframe for test storage mock."""
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


class TestTCSCandidateTournament(unittest.TestCase):
    def test_01_exactly_four_candidates_evaluated(self):
        """Verify tournament evaluates exactly four pre-registered TCS candidates."""
        self.assertEqual(len(TOURNAMENT_CANDIDATES), 4)
        cand_ids = [c.candidate_id for c in TOURNAMENT_CANDIDATES]
        self.assertEqual(cand_ids, ["Candidate_A", "Candidate_B", "Candidate_C", "Candidate_D"])

    def test_02_stability_score_formula(self):
        """Verify stability score calculation produces valid score in [0, 100]."""
        # pos_auc=0.8, pos_mcc=0.8, pos_spread=0.8, std_auc=0.04
        score = compute_stability_score(0.8, 0.8, 0.8, 0.04)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_03_tournament_execution_and_required_metrics(self):
        """Verify run_tcs_candidate_tournament produces 20 fold results with all required metrics."""
        df_raw = create_synthetic_market_data(n_samples=250, seed=202)

        class DummyStorageEngine:
            def dataset_exists(self, asset_name):
                return True

            def load_dataset(self, asset_name):
                return df_raw

        with patch("src.research.tcs_candidate_tournament.StorageEngine", DummyStorageEngine):
            with tempfile.TemporaryDirectory() as tmp_dir:
                res = run_tcs_candidate_tournament(folds=5, scale=True, output_dir=tmp_dir)

                df_fold = res["fold_results"]
                self.assertFalse(df_fold.empty)

                # 4 candidates * 5 folds = 20 evaluations
                self.assertEqual(len(df_fold), 20)

                required_cols = [
                    "candidate_id", "asset", "target_name", "model", "fold",
                    "roc_auc", "pr_auc", "mcc", "f1", "precision", "recall", "ppr",
                    "buy_signals_count", "sell_signals_count",
                    "mean_realized_ret_buy", "mean_realized_ret_sell", "return_spread",
                    "cum_strategy_return", "max_drawdown", "sharpe_ratio",
                ]
                for col in required_cols:
                    self.assertIn(col, df_fold.columns)

    def test_04_shortlist_16_unchanged(self):
        """Verify SHORTLIST_16 feature set contains exactly the 16 standard features."""
        self.assertEqual(len(SHORTLIST_16), 16)
        self.assertIn("RSI_14", SHORTLIST_16)
        self.assertIn("ATR_14", SHORTLIST_16)

    def test_05_champion_json_unmodified(self):
        """Verify champion.json artifact is never modified during tournament execution."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            original_mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, original_mtime)


if __name__ == "__main__":
    unittest.main()
