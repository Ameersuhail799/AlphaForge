"""Mission 13: Robust Dynamic Threshold Experiment Runner.

Evaluates Treatments A, B, C, D, E across assets, models, and outer expanding folds:
- Treatment A: Fixed 0.50 Threshold
- Treatment B: Unconstrained F1 (Mission 12 Control)
- Treatment C: Precision-Constrained F1
- Treatment D: Matthews Correlation Coefficient (MCC)
- Treatment E: Youden's J Statistic (TPR - FPR)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data.storage import StorageEngine
from src.dataset.scaler import FeatureScaler
from src.dataset.target import TargetBuilder
from src.features.feature_pipeline import FeaturePipeline
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
from src.research.robust_threshold_selector import compute_threshold_metrics, select_robust_threshold
from src.research.threshold_selector import select_dynamic_threshold
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ASSETS = ["reliance_ns", "tcs_ns", "hdfcbank_ns", "infy_ns", "icicibank_ns"]

SHORTLIST_16 = [
    "GAP_PCT",
    "OPEN_CLOSE_PCT",
    "HIGH_LOW_PCT",
    "BODY_SIZE",
    "UPPER_WICK",
    "LOWER_WICK",
    "ROC_12",
    "RSI_14",
    "PRICE_CHANGE_PCT",
    "DAILY_RETURN",
    "ROLLING_STD_20",
    "HIST_VOL_20",
    "ATR_14",
    "DAILY_RANGE_PCT",
    "VOLUME_RATIO",
    "VOLUME_CHANGE_PCT",
]


@dataclass
class RobustThresholdFoldResult:
    asset: str
    model: str
    fold: int
    train_samples: int
    validation_samples: int
    base_positive_rate: float
    precision_floor_C: float
    training_time: float
    prediction_time: float
    roc_auc: float
    # Treatment A: Fixed 0.50
    tau_A: float
    accuracy_A: float
    precision_A: float
    recall_A: float
    f1_A: float
    mcc_A: float
    youden_j_A: float
    ppr_A: float
    # Treatment B: Unconstrained F1
    tau_B: float
    accuracy_B: float
    precision_B: float
    recall_B: float
    f1_B: float
    mcc_B: float
    youden_j_B: float
    ppr_B: float
    # Treatment C: Precision-Constrained F1
    tau_C: float
    accuracy_C: float
    precision_C: float
    recall_C: float
    f1_C: float
    mcc_C: float
    youden_j_C: float
    ppr_C: float
    fallback_C: bool
    # Treatment D: MCC Optimization
    tau_D: float
    accuracy_D: float
    precision_D: float
    recall_D: float
    f1_D: float
    mcc_D: float
    youden_j_D: float
    ppr_D: float
    # Treatment E: Youden's J Optimization
    tau_E: float
    accuracy_E: float
    precision_E: float
    recall_E: float
    f1_E: float
    mcc_E: float
    youden_j_E: float
    ppr_E: float


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


def evaluate_val_metrics(y_true: np.ndarray, probs: np.ndarray, tau: float) -> Dict[str, float]:
    """Compute validation metrics for probabilities given threshold tau."""
    m = compute_threshold_metrics(y_true, probs, tau)
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, probs))
        else:
            auc = 0.50
    except Exception:
        auc = 0.50
    m["roc_auc"] = auc
    return m


def run_robust_threshold_experiment(
    assets: Sequence[str] = DEFAULT_ASSETS,
    folds: int = 5,
    scale: bool = True,
    output_dir: Path | str = Path("reports") / "research",
) -> Dict[str, pd.DataFrame]:
    """Execute robust threshold experiment across assets, models, and expanding folds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = StorageEngine()
    fp = FeaturePipeline()
    tb = TargetBuilder()
    registry = ModelRegistry()
    trainer = Trainer()

    models = registry.list_models()
    fold_results: List[RobustThresholdFoldResult] = []

    for asset_name in assets:
        logger.info("Processing asset for Mission 13 experiment: %s", asset_name)
        if not storage.dataset_exists(asset_name):
            logger.warning("Dataset %s not found in storage. Skipping.", asset_name)
            continue

        raw = storage.load_dataset(asset_name)
        df_features = fp.generate(raw.copy())
        df_with_target = tb.build(df_features)

        total_rows = len(df_with_target)
        test_size = max(1, int(total_rows * 0.15))
        non_test = df_with_target.iloc[:-test_size].copy()
        holdout_partition = df_with_target.iloc[-test_size:].copy()

        logger.info(
            "Asset %s: Total rows=%d, non-test=%d, isolated holdout=%d",
            asset_name,
            total_rows,
            len(non_test),
            len(holdout_partition),
        )

        non_test_index = non_test.index
        outer_folds_positions = _create_folds_index(non_test_index, folds)
        feature_cols = [c for c in SHORTLIST_16 if c in df_features.columns]

        for fold_idx, (train_end_pos, val_end_pos) in enumerate(outer_folds_positions, start=1):
            train_end_idx = non_test_index[train_end_pos]
            val_start_idx = non_test_index[train_end_pos + 1]
            val_end_idx = non_test_index[val_end_pos]

            outer_train_df = non_test.loc[:train_end_idx]
            outer_val_df = non_test.loc[val_start_idx:val_end_idx]

            outer_train_df = outer_train_df.replace([np.inf, -np.inf], np.nan)
            outer_val_df = outer_val_df.replace([np.inf, -np.inf], np.nan)
            required = list(feature_cols) + [tb.DEFAULT_TARGET]

            outer_train_df = outer_train_df.dropna(subset=required).copy()
            outer_val_df = outer_val_df.dropna(subset=required).copy()

            if outer_train_df.empty or outer_val_df.empty:
                logger.warning("Skipping asset %s fold %d: empty after dropna", asset_name, fold_idx)
                continue

            outer_X_train = outer_train_df[feature_cols].copy()
            outer_y_train = outer_train_df[tb.DEFAULT_TARGET]
            outer_X_val = outer_val_df[feature_cols].copy()
            outer_y_val = outer_val_df[tb.DEFAULT_TARGET]

            base_pos_rate = float(np.mean(outer_y_train.values))

            for model_name in models:
                t0_tr = time.perf_counter()
                outer_scaler = FeatureScaler(scale=scale)
                outer_X_train_scaled = outer_scaler.fit_transform_train(outer_X_train)
                outer_X_val_scaled = outer_scaler.transform(outer_X_val)

                outer_model = registry.create(model_name)
                outer_train_bundle = type(
                    "OuterTrainBundle",
                    (),
                    {
                        "X_train": outer_X_train_scaled,
                        "y_train": outer_y_train,
                        "feature_names": feature_cols,
                    },
                )()

                trainer.train(outer_model, outer_train_bundle)
                t_tr = time.perf_counter() - t0_tr

                t0_pred = time.perf_counter()
                outer_val_bundle = type(
                    "OuterValBundle",
                    (),
                    {
                        "X_test": outer_X_val_scaled,
                        "y_test": outer_y_val,
                        "feature_names": feature_cols,
                    },
                )()

                probs = outer_model.predict_proba(outer_val_bundle)
                val_probs = probs[:, 1] if (probs.ndim == 2 and probs.shape[1] == 2) else probs.ravel()
                t_pred = time.perf_counter() - t0_pred

                y_val_arr = outer_y_val.to_numpy().astype(int)

                # Treatment A: Fixed 0.50
                mA = evaluate_val_metrics(y_val_arr, val_probs, 0.50)
                auc_val = mA["roc_auc"]

                # Treatment B: Unconstrained F1 (Mission 12 Control)
                tau_B, _ = select_dynamic_threshold(
                    outer_X_train=outer_X_train,
                    outer_y_train=outer_y_train,
                    model_name=model_name,
                    feature_cols=feature_cols,
                    inner_folds=3,
                    min_recall_floor=0.35,
                )
                mB = evaluate_val_metrics(y_val_arr, val_probs, tau_B)

                # Treatment C: Precision-Constrained F1
                tau_C, meta_C = select_robust_threshold(
                    outer_X_train=outer_X_train,
                    outer_y_train=outer_y_train,
                    model_name=model_name,
                    feature_cols=feature_cols,
                    objective_type="precision_constrained_f1",
                    inner_folds=3,
                )
                mC = evaluate_val_metrics(y_val_arr, val_probs, tau_C)

                # Treatment D: MCC Optimization
                tau_D, _ = select_robust_threshold(
                    outer_X_train=outer_X_train,
                    outer_y_train=outer_y_train,
                    model_name=model_name,
                    feature_cols=feature_cols,
                    objective_type="mcc",
                    inner_folds=3,
                )
                mD = evaluate_val_metrics(y_val_arr, val_probs, tau_D)

                # Treatment E: Youden's J Optimization
                tau_E, _ = select_robust_threshold(
                    outer_X_train=outer_X_train,
                    outer_y_train=outer_y_train,
                    model_name=model_name,
                    feature_cols=feature_cols,
                    objective_type="youden_j",
                    inner_folds=3,
                )
                mE = evaluate_val_metrics(y_val_arr, val_probs, tau_E)

                fold_results.append(
                    RobustThresholdFoldResult(
                        asset=asset_name,
                        model=model_name,
                        fold=fold_idx,
                        train_samples=len(outer_X_train_scaled),
                        validation_samples=len(outer_X_val_scaled),
                        base_positive_rate=base_pos_rate,
                        precision_floor_C=meta_C.get("precision_floor", base_pos_rate + 0.025),
                        training_time=t_tr,
                        prediction_time=t_pred,
                        roc_auc=auc_val,
                        tau_A=0.50,
                        accuracy_A=mA["accuracy"],
                        precision_A=mA["precision"],
                        recall_A=mA["recall"],
                        f1_A=mA["f1"],
                        mcc_A=mA["mcc"],
                        youden_j_A=mA["youden_j"],
                        ppr_A=mA["ppr"],
                        tau_B=tau_B,
                        accuracy_B=mB["accuracy"],
                        precision_B=mB["precision"],
                        recall_B=mB["recall"],
                        f1_B=mB["f1"],
                        mcc_B=mB["mcc"],
                        youden_j_B=mB["youden_j"],
                        ppr_B=mB["ppr"],
                        tau_C=tau_C,
                        accuracy_C=mC["accuracy"],
                        precision_C=mC["precision"],
                        recall_C=mC["recall"],
                        f1_C=mC["f1"],
                        mcc_C=mC["mcc"],
                        youden_j_C=mC["youden_j"],
                        ppr_C=mC["ppr"],
                        fallback_C=meta_C.get("fallback_used", False),
                        tau_D=tau_D,
                        accuracy_D=mD["accuracy"],
                        precision_D=mD["precision"],
                        recall_D=mD["recall"],
                        f1_D=mD["f1"],
                        mcc_D=mD["mcc"],
                        youden_j_D=mD["youden_j"],
                        ppr_D=mD["ppr"],
                        tau_E=tau_E,
                        accuracy_E=mE["accuracy"],
                        precision_E=mE["precision"],
                        recall_E=mE["recall"],
                        f1_E=mE["f1"],
                        mcc_E=mE["mcc"],
                        youden_j_E=mE["youden_j"],
                        ppr_E=mE["ppr"],
                    )
                )

    df_fold = pd.DataFrame([asdict(r) for r in fold_results])
    df_summary = _compute_objective_summary(df_fold)
    df_by_objective = df_summary.copy()
    df_by_asset = _compute_asset_summary(df_fold)

    df_fold.to_csv(output_dir / "robust_threshold_fold_results.csv", index=False)
    df_summary.to_csv(output_dir / "robust_threshold_summary.csv", index=False)
    df_by_objective.to_csv(output_dir / "robust_threshold_by_objective.csv", index=False)
    df_by_asset.to_csv(output_dir / "robust_threshold_by_asset.csv", index=False)

    _write_markdown_report(output_dir / "MISSION_13_ROBUST_THRESHOLD_REPORT.md", df_fold, df_summary, df_by_asset)

    return {
        "fold_results": df_fold,
        "summary": df_summary,
        "by_objective": df_by_objective,
        "by_asset": df_by_asset,
    }


def _compute_objective_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate summary metrics for Treatments A, B, C, D, E."""
    if df_fold.empty:
        return pd.DataFrame()

    total_folds = len(df_fold)
    treatments = [
        ("A", "Fixed 0.50"),
        ("B", "Unconstrained F1 (Control)"),
        ("C", "Precision-Constrained F1"),
        ("D", "MCC Optimization"),
        ("E", "Youden's J Optimization"),
    ]

    records = []
    for code, name in treatments:
        tau_col = f"tau_{code}"
        prec_col = f"precision_{code}"
        rec_col = f"recall_{code}"
        f1_col = f"f1_{code}"
        mcc_col = f"mcc_{code}"
        yj_col = f"youden_j_{code}"
        ppr_col = f"ppr_{code}"

        mean_tau = float(df_fold[tau_col].mean())
        std_tau = float(df_fold[tau_col].std())
        mean_prec = float(df_fold[prec_col].mean())
        mean_rec = float(df_fold[rec_col].mean())
        mean_f1 = float(df_fold[f1_col].mean())
        mean_mcc = float(df_fold[mcc_col].mean())
        mean_yj = float(df_fold[yj_col].mean())
        mean_ppr = float(df_fold[ppr_col].mean())

        recall_failures = int((df_fold[rec_col] < 0.25).sum())
        degenerate_ppr = int(((df_fold[ppr_col] > 0.80) | (df_fold[ppr_col] < 0.20)).sum())

        records.append({
            "treatment": code,
            "objective_name": name,
            "total_folds": total_folds,
            "mean_threshold": mean_tau,
            "std_threshold": std_tau,
            "mean_ppr": mean_ppr,
            "degenerate_ppr_folds": degenerate_ppr,
            "mean_precision": mean_prec,
            "mean_recall": mean_rec,
            "recall_failures_below_25": recall_failures,
            "mean_f1": mean_f1,
            "mean_mcc": mean_mcc,
            "mean_youden_j": mean_yj,
        })

    return pd.DataFrame(records)


def _compute_asset_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute per-asset summary metrics across treatments."""
    records = []
    for asset_name, grp in df_fold.groupby("asset"):
        total = len(grp)
        records.append({
            "asset": asset_name,
            "total_folds": total,
            "mean_prec_A": float(grp["precision_A"].mean()),
            "mean_prec_C": float(grp["precision_C"].mean()),
            "mean_prec_D": float(grp["precision_D"].mean()),
            "mean_rec_A": float(grp["recall_A"].mean()),
            "mean_rec_C": float(grp["recall_C"].mean()),
            "mean_rec_D": float(grp["recall_D"].mean()),
            "mean_ppr_A": float(grp["ppr_A"].mean()),
            "mean_ppr_B": float(grp["ppr_B"].mean()),
            "mean_ppr_C": float(grp["ppr_C"].mean()),
            "mean_ppr_D": float(grp["ppr_D"].mean()),
            "mean_ppr_E": float(grp["ppr_E"].mean()),
            "mean_mcc_D": float(grp["mcc_D"].mean()),
            "mean_youden_j_E": float(grp["youden_j_E"].mean()),
        })
    return pd.DataFrame(records)


def _write_markdown_report(
    filepath: Path,
    df_fold: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_by_asset: pd.DataFrame,
) -> None:
    lines = [
        "# Mission 13: Robust Dynamic Threshold Experiment Report",
        "",
        f"**Total Outer Evaluations:** {len(df_fold)} across 5 assets, 3 models, and 5 outer expanding folds",
        "",
        "## Executive Summary",
        "",
        "This experiment evaluates 5 decision threshold selection strategies (Treatments A through E) on `SHORTLIST_16` across 5 liquid NSE assets to eliminate model collapse while preventing degenerate positive prediction behavior.",
        "",
        "## Treatment Summary Table",
        "",
    ]

    if not df_summary.empty:
        cols = list(df_summary.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_summary.iterrows():
            vals = [f"{r[c]:.4f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("## Per-Asset Breakdown")
    lines.append("")
    if not df_by_asset.empty:
        cols = list(df_by_asset.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_by_asset.iterrows():
            vals = [f"{r[c]:.4f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("## Complete Fold Evaluations")
    lines.append("")
    if not df_fold.empty:
        cols = ["asset", "model", "fold", "tau_A", "tau_B", "tau_C", "tau_D", "tau_E", "ppr_A", "ppr_B", "ppr_C", "ppr_D", "ppr_E", "mcc_D", "youden_j_E"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_fold.iterrows():
            vals = [f"{r[c]:.4f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")

    filepath.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.perf_counter()
    logger.info("Executing Mission 13 Full Robust Threshold Experiment...")
    res = run_robust_threshold_experiment()
    elapsed = time.perf_counter() - t0
    logger.info("Mission 13 Full Experiment completed in %.2f seconds.", elapsed)
