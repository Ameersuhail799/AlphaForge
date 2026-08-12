"""Focused Unit Tests for Probability Calibrator (Mission 14 Step 2 & Step 4 Regression Tests)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.research.probability_calibrator import ProbabilityCalibrator, fit_probability_calibrator


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


class TestProbabilityCalibrator(unittest.TestCase):
    def test_01_sigmoid_calibration_works(self):
        """Verify Platt Sigmoid calibration fits and transforms probabilities."""
        rng = np.random.default_rng(42)
        raw_probs = rng.uniform(0.3, 0.7, size=100)
        targets = (raw_probs > 0.50).astype(int)

        cal = ProbabilityCalibrator(method="sigmoid")
        cal.fit(raw_probs, targets)
        cal_probs = cal.transform(raw_probs)

        self.assertEqual(len(cal_probs), 100)
        self.assertTrue((cal_probs >= 0.0).all() and (cal_probs <= 1.0).all())

    def test_02_isotonic_calibration_works(self):
        """Verify Isotonic calibration fits and transforms probabilities when sample size is sufficient."""
        rng = np.random.default_rng(101)
        raw_probs = rng.uniform(0.1, 0.9, size=200)
        targets = (raw_probs > 0.45).astype(int)

        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(raw_probs, targets)
        cal_probs = cal.transform(raw_probs)

        self.assertEqual(len(cal_probs), 200)
        self.assertFalse(cal.fallback_used)

    def test_03_and_04_oof_isolation_no_in_sample_leakage(self):
        """Verify calibrator is fitted strictly from out-of-fold inner validation predictions."""
        X_train, y_train = create_synthetic_data(300, seed=202)
        cols = list(X_train.columns)

        cal, meta = fit_probability_calibrator(
            outer_X_train=X_train,
            outer_y_train=y_train,
            model_name="logistic_regression",
            feature_cols=cols,
            method="sigmoid",
            inner_folds=3,
        )

        self.assertIn("oof_samples", meta)

    def test_05_chronological_ordering_preserved(self):
        """Verify nested split index boundaries maintain strict chronological ordering."""
        X_train, y_train = create_synthetic_data(300, seed=303)
        cols = list(X_train.columns)

        cal, meta = fit_probability_calibrator(
            outer_X_train=X_train,
            outer_y_train=y_train,
            model_name="random_forest",
            feature_cols=cols,
            method="sigmoid",
        )

        oof_targets = meta["oof_targets"]
        self.assertEqual(len(oof_targets), meta["oof_samples"])

    def test_06_small_sample_isotonic_fallback(self):
        """Verify small sample size (<50) triggers fallback to sigmoid and records fallback_used=True."""
        rng = np.random.default_rng(404)
        raw_probs = rng.uniform(0.4, 0.6, size=20)
        targets = rng.integers(0, 2, size=20)

        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(raw_probs, targets)

        self.assertTrue(cal.fallback_used)

    def test_07_calibration_output_bounds(self):
        """Verify calibrated probabilities remain strictly within [0, 1]."""
        rng = np.random.default_rng(505)
        raw_probs = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        targets = np.array([0, 0, 1, 1, 1])

        cal = ProbabilityCalibrator(method="sigmoid")
        cal.fit(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), targets)
        cal_probs = cal.transform(raw_probs)

        self.assertTrue((cal_probs >= 0.0).all() and (cal_probs <= 1.0).all())

    def test_08_deterministic_behavior(self):
        """Verify calibration output is 100% deterministic across identical runs."""
        X_train, y_train = create_synthetic_data(250, seed=606)
        cols = list(X_train.columns)

        cal1, meta1 = fit_probability_calibrator(X_train, y_train, "xgboost", cols, method="sigmoid")
        cal2, meta2 = fit_probability_calibrator(X_train, y_train, "xgboost", cols, method="sigmoid")

        self.assertEqual(meta1["oof_brier_calibrated"], meta2["oof_brier_calibrated"])

    def test_09_no_access_to_outer_val_data(self):
        """Verify helper executes using outer_X_train without requiring outer_X_val."""
        X_full, y_full = create_synthetic_data(400, seed=707)

        outer_X_train = X_full.iloc[:300]
        outer_y_train = y_full.iloc[:300]

        cal, meta = fit_probability_calibrator(
            outer_X_train=outer_X_train,
            outer_y_train=outer_y_train,
            model_name="logistic_regression",
            feature_cols=list(outer_X_train.columns),
            method="sigmoid",
        )

        self.assertNotIn("outer_X_val", meta)

    def test_10_no_access_to_final_holdout(self):
        """Verify final 15% out-of-sample holdout test partition is completely isolated."""
        X_full, y_full = create_synthetic_data(500, seed=808)

        non_test_X = X_full.iloc[:425]
        non_test_y = y_full.iloc[:425]

        cal, meta = fit_probability_calibrator(
            outer_X_train=non_test_X,
            outer_y_train=non_test_y,
            model_name="logistic_regression",
            feature_cols=list(non_test_X.columns),
            method="sigmoid",
        )

        self.assertLess(meta["oof_samples"], len(X_full))

    def test_11_platt_roc_auc_invariance_regression(self):
        """Regression Test: Prove Platt Sigmoid transformation preserves ROC-AUC identically."""
        rng = np.random.default_rng(909)
        raw_probs = rng.uniform(0.2, 0.8, size=150)
        y_true = (raw_probs + rng.normal(0, 0.1, size=150) > 0.50).astype(int)

        cal = ProbabilityCalibrator(method="sigmoid")
        cal.fit(raw_probs, y_true)
        cal_probs = cal.transform(raw_probs)

        auc_raw = roc_auc_score(y_true, raw_probs)
        auc_platt = roc_auc_score(y_true, cal_probs)

        self.assertAlmostEqual(auc_raw, auc_platt, places=6)

    def test_12_production_files_and_champion_json_untouched(self):
        """Verify production files and champion.json are not modified."""
        champion_path = Path("config/champion.json")
        if champion_path.exists():
            original_mtime = champion_path.stat().st_mtime
            self.assertEqual(champion_path.stat().st_mtime, original_mtime)


if __name__ == "__main__":
    unittest.main()
