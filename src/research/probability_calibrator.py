"""Probability Calibration Helper Module for AlphaForge (Mission 14).

Implements leak-safe Platt (Sigmoid) Scaling and Isotonic Regression probability
calibration fitted exclusively on inner out-of-fold chronological training predictions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.dataset.scaler import FeatureScaler
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _create_folds_index(index: pd.Index, folds: int) -> List[Tuple[int, int]]:
    """Generate expanding-window fold split indices."""
    total = len(index)
    if folds < 1:
        raise ValueError("folds must be >= 1")

    val_size = max(1, total // (folds + 1))
    initial_train = total - folds * val_size

    positions: List[Tuple[int, int]] = []
    for i in range(folds):
        train_end_pos = initial_train + i * val_size - 1
        val_end_pos = train_end_pos + val_size
        positions.append((train_end_pos, val_end_pos))

    return [p for p in positions if 0 <= p[0] < total and 0 <= p[1] < total]


class ProbabilityCalibrator:
    """Fitted probability calibrator wrapping Sigmoid or Isotonic transformation."""

    def __init__(self, method: str = "sigmoid") -> None:
        """Initialize calibrator method.

        Args:
            method: Calibration method ('sigmoid' or 'isotonic').
        """
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"Unsupported calibration method: {method}")

        self.method = method
        self._model: Any = None
        self._is_fitted = False
        self.fallback_used = False

    def fit(self, oof_probs: np.ndarray, oof_targets: np.ndarray) -> ProbabilityCalibrator:
        """Fit calibrator on out-of-fold probabilities and target labels.

        Args:
            oof_probs: Uncalibrated out-of-fold probability predictions.
            oof_targets: True binary targets (0 or 1).

        Returns:
            Fitted ProbabilityCalibrator instance.
        """
        if len(oof_probs) == 0 or len(oof_targets) == 0:
            raise ValueError("oof_probs and oof_targets cannot be empty.")

        probs_arr = np.asarray(oof_probs, dtype=float).ravel()
        targets_arr = np.asarray(oof_targets, dtype=int).ravel()

        if self.method == "isotonic":
            # Small sample or low unique values check for isotonic stability
            if len(probs_arr) < 50 or len(np.unique(probs_arr)) < 5:
                logger.warning(
                    "Small sample size (%d) or insufficient unique probabilities for isotonic calibration. Falling back to sigmoid.",
                    len(probs_arr),
                )
                self.fallback_used = True
                self._model = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
                self._model.fit(probs_arr.reshape(-1, 1), targets_arr)
            else:
                self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                self._model.fit(probs_arr, targets_arr)
        else:
            self._model = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
            self._model.fit(probs_arr.reshape(-1, 1), targets_arr)

        self._is_fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Transform uncalibrated probabilities into calibrated probabilities.

        Args:
            probs: Uncalibrated probability array.

        Returns:
            Calibrated probability array clipped to [0, 1].
        """
        if not self._is_fitted:
            raise RuntimeError("ProbabilityCalibrator must be fitted before calling transform.")

        probs_arr = np.asarray(probs, dtype=float)

        if isinstance(self._model, IsotonicRegression):
            calibrated = self._model.transform(probs_arr.ravel())
        else:
            calibrated = self._model.predict_proba(probs_arr.ravel().reshape(-1, 1))[:, 1]

        calibrated = np.clip(calibrated, 0.0, 1.0)
        return calibrated.reshape(probs_arr.shape)


def fit_probability_calibrator(
    outer_X_train: pd.DataFrame,
    outer_y_train: pd.Series,
    model_name: str,
    feature_cols: Sequence[str],
    method: str = "sigmoid",
    inner_folds: int = 3,
) -> Tuple[ProbabilityCalibrator, Dict[str, Any]]:
    """Fit a leak-safe probability calibrator using nested chronological CV on outer training data only.

    Args:
        outer_X_train: Outer fold training feature matrix.
        outer_y_train: Outer fold training target series.
        model_name: Name of registered model architecture.
        feature_cols: List of feature column names.
        method: Calibration method ('sigmoid' or 'isotonic').
        inner_folds: Number of inner chronological expanding window folds.

    Returns:
        Tuple of (fitted_calibrator_obj, metadata_dict).
    """
    if len(outer_X_train) == 0 or len(outer_y_train) == 0:
        raise ValueError("outer_X_train and outer_y_train cannot be empty.")

    registry = ModelRegistry()
    trainer = Trainer()

    train_df = outer_X_train[list(feature_cols)].copy()
    train_df["__target__"] = outer_y_train.values

    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
    if train_df.empty:
        raise ValueError("No valid rows remain in outer_X_train after cleaning NaNs.")

    X_clean = train_df[list(feature_cols)].copy()
    y_clean = train_df["__target__"].astype(int)

    index = X_clean.index
    fold_positions = _create_folds_index(index, inner_folds)

    oof_probs_list: List[np.ndarray] = []
    oof_y_list: List[np.ndarray] = []

    for fold_idx, (tr_end_pos, val_end_pos) in enumerate(fold_positions, start=1):
        tr_end_idx = index[tr_end_pos]
        val_start_idx = index[tr_end_pos + 1]
        val_end_idx = index[val_end_pos]

        in_tr_df = X_clean.loc[:tr_end_idx]
        in_tr_y = y_clean.loc[:tr_end_idx]

        in_val_df = X_clean.loc[val_start_idx:val_end_idx]
        in_val_y = y_clean.loc[val_start_idx:val_end_idx]

        if in_tr_df.empty or in_val_df.empty:
            continue

        # Fresh FeatureScaler fit ONLY on inner training partition
        inner_scaler = FeatureScaler(scale=True)
        in_tr_scaled = inner_scaler.fit_transform_train(in_tr_df)
        in_val_scaled = inner_scaler.transform(in_val_df)

        inner_model = registry.create(model_name)

        in_tr_bundle = type(
            "InnerTrainBundle",
            (),
            {
                "X_train": in_tr_scaled,
                "y_train": in_tr_y,
                "feature_names": list(feature_cols),
            },
        )()

        trainer.train(inner_model, in_tr_bundle)

        in_val_bundle = type(
            "InnerValBundle",
            (),
            {
                "X_test": in_val_scaled,
                "y_test": in_val_y,
                "feature_names": list(feature_cols),
            },
        )()

        probs = inner_model.predict_proba(in_val_bundle)
        if probs.ndim == 2 and probs.shape[1] == 2:
            positive_probs = probs[:, 1]
        else:
            positive_probs = probs.ravel()

        oof_probs_list.append(positive_probs)
        oof_y_list.append(in_val_y.values)

    if not oof_probs_list:
        raise ValueError(f"No inner OOF predictions could be generated for model {model_name}.")

    oof_probs = np.concatenate(oof_probs_list)
    oof_y = np.concatenate(oof_y_list)

    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(oof_probs, oof_y)

    cal_oof_probs = calibrator.transform(oof_probs)

    oof_probs_clipped = np.clip(oof_probs, 1e-15, 1.0 - 1e-15)
    cal_oof_probs_clipped = np.clip(cal_oof_probs, 1e-15, 1.0 - 1e-15)

    brier_raw = float(brier_score_loss(oof_y, oof_probs_clipped))
    brier_cal = float(brier_score_loss(oof_y, cal_oof_probs_clipped))

    logloss_raw = float(log_loss(oof_y, oof_probs_clipped))
    logloss_cal = float(log_loss(oof_y, cal_oof_probs_clipped))

    metadata = {
        "method": method,
        "fallback_used": calibrator.fallback_used,
        "oof_brier_raw": brier_raw,
        "oof_brier_calibrated": brier_cal,
        "oof_log_loss_raw": logloss_raw,
        "oof_log_loss_calibrated": logloss_cal,
        "oof_samples": len(oof_probs),
        "oof_raw_probs": oof_probs,
        "oof_cal_probs": cal_oof_probs,
        "oof_targets": oof_y,
    }

    return calibrator, metadata
