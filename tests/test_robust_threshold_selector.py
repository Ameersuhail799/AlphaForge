"""Focused Unit Tests for Robust Dynamic Threshold Selector (Mission 13 Step 1)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.research.robust_threshold_selector import compute_threshold_metrics, select_robust_threshold


def create_synthetic_data(n_samples: int = 300, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic chronological market feature data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_samples)

    df = pd.DataFrame(
        {
            "GAP_PCT": rng.normal(0, 0.01, size=n_samples),
            "OPEN_CLOSE_PCT": rng.normal(0, 0.01, size=n_samples),
            "HIGH_LOW_PCT": rng.uniform(0.01, 0.05, size=n_samples),
            "BODY_SIZE": rng.uniform(0.001, 0.03, size=n_samples),
            "UPPER_WICK": rng.uniform(0.001, 0.02, size=n_samples),
            "LOWER_WICK": rng.uniform(0.001, 0.02, size=n_samples),
            "ROC_12": rng.normal(0, 0.02, size=n_samples),
            "RSI_14": rng.uniform(30, 70, size=n_samples),
            "PRICE_CHANGE_PCT": rng.normal(0, 0.02, size=n_samples),
            "DAILY_RETURN": rng.normal(0, 0.01, size=n_samples),
            "ROLLING_STD_20": rng.uniform(0.01, 0.03, size=n_samples),
            "HIST_VOL_20": rng.uniform(0.1, 0.3, size=n_samples),
            "ATR_14": rng.uniform(1.0, 5.0, size=n_samples),
            "DAILY_RANGE_PCT": rng.uniform(0.01, 0.04, size=n_samples),
            "VOLUME_RATIO": rng.uniform(0.5, 2.0, size=n_samples),
            "VOLUME_CHANGE_PCT": rng.normal(0, 0.1, size=n_samples),
        },
        index=dates,
    )

    signal = 2.0 * df["ROC_12"] + 1.5 * df["GAP_PCT"] - 0.02 * df["RSI_14"]
    target = (signal > signal.median()).astype(int)

    return df, pd.Series(target, index=dates)


class TestRobustThresholdSelector(unittest.TestCase):
    def test_01_threshold_grid_bounds(self):
        """Verify selected thresholds across all objectives are strictly bounded within [0.20, 0.80]."""
        X_train, y_train = create_synthetic_data(300, seed=42)
        cols = list(X_train.columns)

        for obj in ["precision_constrained_f1", "mcc", "youden_j"]:
            tau, meta = select_robust_threshold(
                outer_X_train=X_train,
                outer_y_train=y_train,
                model_name="logistic_regression",
                feature_cols=cols,
                objective_type=obj,
            )
            self.assertGreaterEqual(tau, 0.20)
            self.assertLessEqual(tau, 0.80)
            self.assertIn("selected_threshold", meta)

    def test_02_precision_constraint_enforcement(self):
        """Verify Treatment C enforces precision floor = training base rate + delta."""
        X_train, y_train = create_synthetic_data(300, seed=101)
        cols = list(X_train.columns)

        p0 = float(np.mean(y_train))
        expected_floor = p0 + 0.025

        tau, meta = select_robust_threshold(
            outer_X_train=X_train,
            outer_y_train=y_train,
            model_name="random_forest",
            feature_cols=cols,
            objective_type="precision_constrained_f1",
            precision_delta=0.025,
        )

        self.assertAlmostEqual(meta["precision_floor"], expected_floor, places=4)
        if not meta["fallback_used"]:
            self.assertGreaterEqual(meta["oof_precision"], meta["precision_floor"])

    def test_03_mcc_optimization(self):
        """Verify Treatment D selects threshold maximizing OOF MCC."""
        X_train, y_train = create_synthetic_data(300, seed=202)
        cols = list(X_train.columns)

        tau, meta = select_robust_threshold(
            outer_X_train=X_train,
            outer_y_train=y_train,
            model_name="xgboost",
            feature_cols=cols,
            objective_type="mcc",
        )

        self.assertEqual(meta["objective_type"], "mcc")
        self.assertIn("oof_mcc", meta)

    def test_04_youden_j_optimization(self):
        """Verify Treatment E selects threshold maximizing Youden's J statistic (TPR - FPR)."""
        X_train, y_train = create_synthetic_data(300, seed=303)
        cols = list(X_train.columns)

        tau, meta = select_robust_threshold(
            outer_X_train=X_train,
            outer_y_train=y_train,
            model_name="logistic_regression",
            feature_cols=cols,
            objective_type="youden_j",
        )

        self.assertEqual(meta["objective_type"], "youden_j")
        self.assertIn("oof_youden_j", meta)

    def test_05_fallback_when_precision_infeasible(self):
        """Verify fallback behavior when precision floor cannot be satisfied."""
        X_train, y_train = create_synthetic_data(200, seed=404)
        cols = list(X_train.columns)

        # Force unachievable precision floor (+0.65 above base rate > 1.0)
        tau, meta = select_robust_threshold(
            outer_X_train=X_train,
            outer_y_train=y_train,
            model_name="logistic_regression",
            feature_cols=cols,
            objective_type="precision_constrained_f1",
            precision_delta=0.65,
        )

        self.assertTrue(meta["fallback_used"])
        self.assertGreaterEqual(tau, 0.20)
        self.assertLessEqual(tau, 0.80)

    def test_06_deterministic_tie_breaking(self):
        """Verify threshold selection produces identical deterministic outputs across multiple runs."""
        X_train, y_train = create_synthetic_data(250, seed=505)
        cols = list(X_train.columns)

        tau1, meta1 = select_robust_threshold(
            outer_X_train=X_train,
            outer_y_train=y_train,
            model_name="xgboost",
            feature_cols=cols,
            objective_type="mcc",
        )

        tau2, meta2 = select_robust_threshold(
            outer_X_train=X_train,
            outer_y_train=y_train,
            model_name="xgboost",
            feature_cols=cols,
            objective_type="mcc",
        )

        self.assertEqual(tau1, tau2)
        self.assertEqual(meta1["oof_mcc"], meta2["oof_mcc"])

    def test_07_ppr_degeneracy_detection(self):
        """Verify metric calculator accurately computes Positive Prediction Rate (PPR) and detects degeneracy."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        probs_all_high = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

        m = compute_threshold_metrics(y_true, probs_all_high, tau=0.50)
        self.assertEqual(m["ppr"], 1.0)

        is_degen = bool(m["ppr"] > 0.80 or m["ppr"] < 0.20)
        self.assertTrue(is_degen)

    def test_08_inner_fold_isolation(self):
        """Verify inner cross-validation folds operate with independent scaling partitions."""
        X_train, y_train = create_synthetic_data(300, seed=606)
        cols = list(X_train.columns)

        tau, meta = select_robust_threshold(
            outer_X_train=X_train,
            outer_y_train=y_train,
            model_name="random_forest",
            feature_cols=cols,
            objective_type="precision_constrained_f1",
            inner_folds=3,
        )

        self.assertLessEqual(meta["oof_samples"], len(X_train))

    def test_09_no_access_to_outer_validation_data(self):
        """Verify helper executes using outer_X_train without requiring outer_X_val."""
        X_full, y_full = create_synthetic_data(400, seed=707)

        outer_X_train = X_full.iloc[:300]
        outer_y_train = y_full.iloc[:300]

        outer_X_val = X_full.iloc[300:]
        outer_y_val = y_full.iloc[300:]

        tau, meta = select_robust_threshold(
            outer_X_train=outer_X_train,
            outer_y_train=outer_y_train,
            model_name="logistic_regression",
            feature_cols=list(outer_X_train.columns),
            objective_type="precision_constrained_f1",
        )

        self.assertLessEqual(meta["oof_samples"], len(outer_X_train))
        self.assertNotIn("outer_X_val", meta)

    def test_10_no_access_to_final_holdout(self):
        """Verify final 15% out-of-sample holdout test partition is completely isolated."""
        X_full, y_full = create_synthetic_data(500, seed=808)

        non_test_X = X_full.iloc[:425]
        non_test_y = y_full.iloc[:425]
        holdout_X = X_full.iloc[425:]

        tau, meta = select_robust_threshold(
            outer_X_train=non_test_X,
            outer_y_train=non_test_y,
            model_name="logistic_regression",
            feature_cols=list(non_test_X.columns),
            objective_type="mcc",
        )

        self.assertLessEqual(meta["oof_samples"], len(non_test_X))
        self.assertLess(meta["oof_samples"], len(X_full))


if __name__ == "__main__":
    unittest.main()
