"""Robust Dynamic Threshold Selector for Mission 13.

Implements leak-safe threshold selection across multiple candidate objectives:
- Treatment C: Precision-Constrained F1
- Treatment D: Matthews Correlation Coefficient (MCC)
- Treatment E: Youden's J Statistic (TPR - FPR)

All threshold selection operates exclusively on inner out-of-fold predictions
generated via nested chronological cross-validation inside outer_X_train.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

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


def compute_threshold_metrics(y_true: np.ndarray, probs: np.ndarray, tau: float) -> Dict[str, float]:
    """Compute classification and diagnostic metrics for a given threshold tau.

    Args:
        y_true: True binary target array (0 or 1).
        probs: Predicted probability array.
        tau: Decision threshold.

    Returns:
        Dictionary of computed metrics (accuracy, precision, recall, f1, mcc, youden_j, ppr).
    """
    preds = (probs >= tau).astype(int)
    n_samples = len(y_true)

    if n_samples == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
            "youden_j": 0.0,
            "ppr": 0.0,
        }

    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    ppr = float(np.mean(preds))

    # Safe MCC calculation
    try:
        if len(np.unique(y_true)) > 1 and len(np.unique(preds)) > 1:
            mcc = float(matthews_corrcoef(y_true, preds))
        else:
            mcc = 0.0
    except Exception:
        mcc = 0.0

    # Youden's J = TPR - FPR
    tp = np.sum((preds == 1) & (y_true == 1))
    fn = np.sum((preds == 0) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    tn = np.sum((preds == 0) & (y_true == 0))

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    youden_j = tpr - fpr

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "youden_j": youden_j,
        "ppr": ppr,
    }


def select_robust_threshold(
    outer_X_train: pd.DataFrame,
    outer_y_train: pd.Series,
    model_name: str,
    feature_cols: Sequence[str],
    objective_type: str = "precision_constrained_f1",
    inner_folds: int = 3,
    precision_delta: float = 0.025,
    min_recall_floor: float = 0.30,
) -> Tuple[float, Dict[str, Any]]:
    """Select robust threshold via nested chronological CV on outer training data only.

    Args:
        outer_X_train: Outer fold training feature matrix.
        outer_y_train: Outer fold training target series.
        model_name: Name of registered model to instantiate.
        feature_cols: List of feature column names to evaluate.
        objective_type: Objective to optimize ('precision_constrained_f1', 'mcc', 'youden_j').
        inner_folds: Number of inner chronological expanding window folds.
        precision_delta: Margin above training base rate for precision floor (default +0.025).
        min_recall_floor: Absolute minimum recall floor for precision-constrained search.

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

    base_rate_p0 = float(np.mean(y_clean.values))
    precision_floor = base_rate_p0 + precision_delta

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

        # FeatureScaler fitted ONLY on inner training partition
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
        logger.warning("No inner OOF predictions generated. Defaulting to 0.50 threshold.")
        return 0.50, {
            "selected_threshold": 0.50,
            "fallback_used": True,
            "reason": "no_oof_predictions",
            "base_rate_p0": base_rate_p0,
            "precision_floor": precision_floor,
        }

    oof_probs = np.concatenate(oof_probs_list)
    oof_y = np.concatenate(oof_y_list)

    # Threshold search grid: 0.20 to 0.80 in steps of 0.01
    candidate_thresholds = np.round(np.arange(0.20, 0.805, 0.01), 2)

    eval_records = []
    for tau in candidate_thresholds:
        m = compute_threshold_metrics(oof_y, oof_probs, float(tau))
        dist_05 = abs(float(tau) - 0.50)
        eval_records.append({
            "tau": float(tau),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "mcc": m["mcc"],
            "youden_j": m["youden_j"],
            "ppr": m["ppr"],
            "dist_05": dist_05,
        })

    df_eval = pd.DataFrame(eval_records)
    fallback_used = False

    if objective_type == "precision_constrained_f1":
        feasible = df_eval[(df_eval["precision"] >= precision_floor) & (df_eval["recall"] >= min_recall_floor)]
        if not feasible.empty:
            max_val = feasible["f1"].max()
            candidates = feasible[np.isclose(feasible["f1"], max_val, atol=1e-9)]
        else:
            fallback_used = True
            # Fallback 1: Filter candidates with recall >= min_recall_floor and maximize precision
            rec_feasible = df_eval[df_eval["recall"] >= min_recall_floor]
            if not rec_feasible.empty:
                max_p = rec_feasible["precision"].max()
                candidates = rec_feasible[np.isclose(rec_feasible["precision"], max_p, atol=1e-9)]
            else:
                # Fallback 2: Maximize MCC
                max_mcc = df_eval["mcc"].max()
                candidates = df_eval[np.isclose(df_eval["mcc"], max_mcc, atol=1e-9)]

    elif objective_type == "mcc":
        max_val = df_eval["mcc"].max()
        candidates = df_eval[np.isclose(df_eval["mcc"], max_val, atol=1e-9)]

    elif objective_type == "youden_j":
        max_val = df_eval["youden_j"].max()
        candidates = df_eval[np.isclose(df_eval["youden_j"], max_val, atol=1e-9)]

    else:
        raise ValueError(f"Unknown objective_type: {objective_type}")

    # Deterministic tie-breaking: sort by distance to 0.50, then tau
    candidates = candidates.sort_values(by=["dist_05", "tau"])
    best_row = candidates.iloc[0]

    selected_tau = float(best_row["tau"])
    is_degenerate_ppr = bool(best_row["ppr"] > 0.80 or best_row["ppr"] < 0.20)

    metadata = {
        "selected_threshold": selected_tau,
        "objective_type": objective_type,
        "oof_precision": float(best_row["precision"]),
        "oof_recall": float(best_row["recall"]),
        "oof_f1": float(best_row["f1"]),
        "oof_mcc": float(best_row["mcc"]),
        "oof_youden_j": float(best_row["youden_j"]),
        "oof_ppr": float(best_row["ppr"]),
        "base_rate_p0": base_rate_p0,
        "precision_floor": precision_floor,
        "fallback_used": fallback_used,
        "is_degenerate_ppr": is_degenerate_ppr,
        "oof_samples": len(oof_probs),
    }

    return selected_tau, metadata
