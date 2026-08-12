"""Nested Leak-Safe Dynamic Threshold Selection Helper for AlphaForge.

Performs nested chronological cross-validation inside outer training data
to select an optimal decision threshold without validation or test leakage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

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


def select_dynamic_threshold(
    outer_X_train: pd.DataFrame,
    outer_y_train: pd.Series,
    model_name: str,
    feature_cols: Sequence[str],
    inner_folds: int = 3,
    min_recall_floor: float = 0.35,
) -> Tuple[float, Dict[str, Any]]:
    """Select dynamic decision threshold via nested chronological CV on training data only.

    Args:
        outer_X_train: Outer fold training feature matrix.
        outer_y_train: Outer fold training target series.
        model_name: Name of registered model to create and train.
        feature_cols: List of feature column names to evaluate.
        inner_folds: Number of inner chronological expanding window folds.
        min_recall_floor: Required OOF recall threshold floor.

    Returns:
        Tuple of (selected_threshold, metadata_dict).
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
    scaler_histories: List[Dict[str, Any]] = []

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

        scaler_histories.append({
            "fold": fold_idx,
            "mean": inner_scaler._mean.to_dict() if inner_scaler._mean is not None else {},
            "scale": inner_scaler._scale.to_dict() if inner_scaler._scale is not None else {},
        })

        # Fresh model instance
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
        logger.warning("No inner OOF predictions generated. Defaulting to 0.50 threshold.")
        return 0.50, {"fallback_used": True, "reason": "no_oof_predictions"}

    oof_probs = np.concatenate(oof_probs_list)
    oof_y = np.concatenate(oof_y_list)

    # Grid search threshold from 0.20 through 0.80 in steps of 0.01
    candidate_thresholds = np.round(np.arange(0.20, 0.805, 0.01), 2)

    eval_records = []
    for tau in candidate_thresholds:
        preds = (oof_probs >= tau).astype(int)
        rec = float(recall_score(oof_y, preds, zero_division=0))
        f1 = float(f1_score(oof_y, preds, zero_division=0))
        dist_05 = abs(tau - 0.50)
        eval_records.append({
            "tau": float(tau),
            "recall": rec,
            "f1": f1,
            "dist_05": dist_05,
        })

    df_eval = pd.DataFrame(eval_records)
    feasible = df_eval[df_eval["recall"] >= min_recall_floor]

    if not feasible.empty:
        max_f1 = feasible["f1"].max()
        best_candidates = feasible[np.isclose(feasible["f1"], max_f1, atol=1e-9)]
        best_candidates = best_candidates.sort_values(by=["dist_05", "tau"])
        selected_tau = float(best_candidates["tau"].iloc[0])
        selected_rec = float(best_candidates["recall"].iloc[0])
        selected_f1 = float(best_candidates["f1"].iloc[0])
        fallback_used = False
    else:
        max_rec = df_eval["recall"].max()
        best_candidates = df_eval[np.isclose(df_eval["recall"], max_rec, atol=1e-9)]
        max_f1_in_rec = best_candidates["f1"].max()
        best_candidates = best_candidates[np.isclose(best_candidates["f1"], max_f1_in_rec, atol=1e-9)]
        best_candidates = best_candidates.sort_values(by=["dist_05", "tau"])
        selected_tau = float(best_candidates["tau"].iloc[0])
        selected_rec = float(best_candidates["recall"].iloc[0])
        selected_f1 = float(best_candidates["f1"].iloc[0])
        fallback_used = True

    metadata = {
        "selected_threshold": selected_tau,
        "oof_recall": selected_rec,
        "oof_f1": selected_f1,
        "feasible_count": len(feasible),
        "fallback_used": fallback_used,
        "oof_samples": len(oof_probs),
        "scaler_histories": scaler_histories,
    }

    return selected_tau, metadata
