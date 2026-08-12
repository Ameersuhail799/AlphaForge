"""Mission 14: Probability Calibration Experiment Runner.

Evaluates Treatment A (Uncalibrated), Treatment B (Platt/Sigmoid Calibration),
and Treatment C (Isotonic Calibration) across 5 assets, 3 models, and 5 outer expanding folds.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.data.storage import StorageEngine
from src.dataset.scaler import FeatureScaler
from src.dataset.target import TargetBuilder
from src.features.feature_pipeline import FeaturePipeline
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
from src.research.probability_calibrator import ProbabilityCalibrator, fit_probability_calibrator
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
class CalibrationFoldResult:
    asset: str
    model: str
    fold: int
    train_samples: int
    validation_samples: int
    base_positive_rate: float
    training_time: float
    prediction_time: float
    # Treatment A: Uncalibrated
    brier_A: float
    log_loss_A: float
    ece_A: float
    roc_auc_A: float
    ppr_050_A: float
    # Treatment B: Sigmoid / Platt
    brier_B: float
    log_loss_B: float
    ece_B: float
    roc_auc_B: float
    ppr_050_B: float
    rank_corr_B: float
    rank_inversions_B: int
    fallback_B: bool
    calibration_time_B: float
    # Treatment C: Isotonic
    brier_C: float
    log_loss_C: float
    ece_C: float
    roc_auc_C: float
    ppr_050_C: float
    rank_corr_C: float
    rank_inversions_C: int
    fallback_C: bool
    calibration_time_C: float


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


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across n_bins probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    probs_arr = np.asarray(probs, dtype=float)
    y_arr = np.asarray(y_true, dtype=int)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        if i == 0:
            mask = (probs_arr >= bin_lower) & (probs_arr <= bin_upper)
        else:
            mask = (probs_arr > bin_lower) & (probs_arr <= bin_upper)

        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_arr[mask]))
            bin_conf = float(np.mean(probs_arr[mask]))
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def count_pairwise_rank_inversions(raw_probs: np.ndarray, cal_probs: np.ndarray) -> int:
    """Count pairwise rank inversions where relative rank order is flipped."""
    raw_arr = np.asarray(raw_probs, dtype=float)
    cal_arr = np.asarray(cal_probs, dtype=float)
    diff_raw = np.subtract.outer(raw_arr, raw_arr)
    diff_cal = np.subtract.outer(cal_arr, cal_arr)
    inversions = np.sum((diff_raw * diff_cal) < 0) // 2
    return int(inversions)


def evaluate_calibration_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    raw_probs: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute probability calibration metrics without silent fallbacks."""
    probs_clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
    y_arr = np.asarray(y_true, dtype=int)

    if len(np.unique(y_arr)) < 2:
        raise ValueError("ROC-AUC cannot be calculated because test target partition has only 1 class.")

    brier = float(brier_score_loss(y_arr, probs_clipped))
    ll = float(log_loss(y_arr, probs_clipped))
    ece = compute_ece(y_arr, probs_clipped, n_bins=10)
    auc = float(roc_auc_score(y_arr, probs_clipped))
    ppr_050 = float(np.mean((probs_clipped >= 0.50).astype(int)))

    metrics = {
        "brier": brier,
        "log_loss": ll,
        "ece": ece,
        "roc_auc": auc,
        "ppr_050": ppr_050,
    }

    if raw_probs is not None:
        raw_arr = np.asarray(raw_probs, dtype=float)
        try:
            rho, _ = spearmanr(raw_arr, probs_clipped)
            if np.isnan(rho):
                rho = 1.0 if np.allclose(raw_arr, probs_clipped) else 0.0
        except Exception:
            rho = 1.0
        metrics["rank_corr"] = float(rho)
        metrics["rank_inversions"] = count_pairwise_rank_inversions(raw_arr, probs_clipped)
    else:
        metrics["rank_corr"] = 1.0
        metrics["rank_inversions"] = 0

    return metrics


def run_calibration_experiment(
    assets: Sequence[str] = DEFAULT_ASSETS,
    folds: int = 5,
    scale: bool = True,
    output_dir: Path | str = Path("reports") / "research",
) -> Dict[str, pd.DataFrame]:
    """Execute full calibration comparison experiment across assets, models, and outer folds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = StorageEngine()
    fp = FeaturePipeline()
    tb = TargetBuilder()
    registry = ModelRegistry()
    trainer = Trainer()

    models = registry.list_models()
    fold_results: List[CalibrationFoldResult] = []

    for asset_name in assets:
        logger.info("Processing asset for Mission 14 calibration experiment: %s", asset_name)
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

                # Treatment A: Uncalibrated
                mA = evaluate_calibration_metrics(y_val_arr, val_probs, raw_probs=None)

                # Treatment B: Sigmoid / Platt Calibration
                t0_cal_B = time.perf_counter()
                cal_B, meta_B = fit_probability_calibrator(
                    outer_X_train=outer_X_train,
                    outer_y_train=outer_y_train,
                    model_name=model_name,
                    feature_cols=feature_cols,
                    method="sigmoid",
                    inner_folds=3,
                )
                cal_probs_B = cal_B.transform(val_probs)
                t_cal_B = time.perf_counter() - t0_cal_B
                mB = evaluate_calibration_metrics(y_val_arr, cal_probs_B, raw_probs=val_probs)

                # Treatment C: Isotonic Calibration
                t0_cal_C = time.perf_counter()
                cal_C, meta_C = fit_probability_calibrator(
                    outer_X_train=outer_X_train,
                    outer_y_train=outer_y_train,
                    model_name=model_name,
                    feature_cols=feature_cols,
                    method="isotonic",
                    inner_folds=3,
                )
                cal_probs_C = cal_C.transform(val_probs)
                t_cal_C = time.perf_counter() - t0_cal_C
                mC = evaluate_calibration_metrics(y_val_arr, cal_probs_C, raw_probs=val_probs)

                fold_results.append(
                    CalibrationFoldResult(
                        asset=asset_name,
                        model=model_name,
                        fold=fold_idx,
                        train_samples=len(outer_X_train_scaled),
                        validation_samples=len(outer_X_val_scaled),
                        base_positive_rate=base_pos_rate,
                        training_time=t_tr,
                        prediction_time=t_pred,
                        brier_A=mA["brier"],
                        log_loss_A=mA["log_loss"],
                        ece_A=mA["ece"],
                        roc_auc_A=mA["roc_auc"],
                        ppr_050_A=mA["ppr_050"],
                        brier_B=mB["brier"],
                        log_loss_B=mB["log_loss"],
                        ece_B=mB["ece"],
                        roc_auc_B=mB["roc_auc"],
                        ppr_050_B=mB["ppr_050"],
                        rank_corr_B=mB["rank_corr"],
                        rank_inversions_B=int(mB["rank_inversions"]),
                        fallback_B=meta_B.get("fallback_used", False),
                        calibration_time_B=t_cal_B,
                        brier_C=mC["brier"],
                        log_loss_C=mC["log_loss"],
                        ece_C=mC["ece"],
                        roc_auc_C=mC["roc_auc"],
                        ppr_050_C=mC["ppr_050"],
                        rank_corr_C=mC["rank_corr"],
                        rank_inversions_C=int(mC["rank_inversions"]),
                        fallback_C=meta_C.get("fallback_used", False),
                        calibration_time_C=t_cal_C,
                    )
                )

    df_fold = pd.DataFrame([asdict(r) for r in fold_results])
    df_summary = _compute_summary(df_fold)
    df_by_asset = _compute_asset_summary(df_fold)
    df_by_model = _compute_model_summary(df_fold)

    df_fold.to_csv(output_dir / "calibration_results.csv", index=False)
    df_fold.to_csv(output_dir / "calibration_fold_results.csv", index=False)
    df_summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    df_by_asset.to_csv(output_dir / "calibration_by_asset.csv", index=False)
    df_by_model.to_csv(output_dir / "calibration_by_model.csv", index=False)

    _write_markdown_report(output_dir / "CALIBRATION_EXPERIMENT_REPORT.md", df_fold, df_summary, df_by_asset, df_by_model)

    return {
        "fold_results": df_fold,
        "summary": df_summary,
        "by_asset": df_by_asset,
        "by_model": df_by_model,
    }


def _compute_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate summary metrics across Treatments A, B, and C."""
    if df_fold.empty:
        return pd.DataFrame()

    total_folds = len(df_fold)
    treatments = [
        ("A", "Uncalibrated Baseline"),
        ("B", "Platt / Sigmoid Calibration"),
        ("C", "Isotonic Calibration"),
    ]

    records = []
    brier_base = float(df_fold["brier_A"].mean())
    ll_base = float(df_fold["log_loss_A"].mean())
    auc_base = float(df_fold["roc_auc_A"].mean())

    for code, name in treatments:
        brier = float(df_fold[f"brier_{code}"].mean())
        ll = float(df_fold[f"log_loss_{code}"].mean())
        ece = float(df_fold[f"ece_{code}"].mean())
        auc = float(df_fold[f"roc_auc_{code}"].mean())
        ppr = float(df_fold[f"ppr_050_{code}"].mean())

        rank_corr = float(df_fold[f"rank_corr_{code}"].mean()) if f"rank_corr_{code}" in df_fold.columns else 1.0
        inversions = float(df_fold[f"rank_inversions_{code}"].mean()) if f"rank_inversions_{code}" in df_fold.columns else 0.0

        brier_red = brier_base - brier
        ll_red = ll_base - ll
        auc_loss = auc_base - auc

        degen_ppr = int(((df_fold[f"ppr_050_{code}"] > 0.80) | (df_fold[f"ppr_050_{code}"] < 0.20)).sum())
        fallback_cnt = int(df_fold[f"fallback_{code}"].sum()) if f"fallback_{code}" in df_fold.columns else 0

        records.append({
            "treatment": code,
            "method_name": name,
            "total_folds": total_folds,
            "mean_brier": brier,
            "brier_reduction": brier_red,
            "mean_log_loss": ll,
            "log_loss_reduction": ll_red,
            "mean_ece": ece,
            "mean_roc_auc": auc,
            "roc_auc_degradation": auc_loss,
            "mean_ppr_050": ppr,
            "mean_rank_correlation": rank_corr,
            "mean_rank_inversions": inversions,
            "degenerate_ppr_folds": degen_ppr,
            "fallback_count": fallback_cnt,
        })

    df_sum = pd.DataFrame(records)

    # Acceptance Criteria Verification for Treatment B & C
    row_B = df_sum[df_sum["treatment"] == "B"].iloc[0]
    row_C = df_sum[df_sum["treatment"] == "C"].iloc[0]

    crit1_B = bool(row_B["brier_reduction"] >= 0.005)
    crit2_B = bool(row_B["log_loss_reduction"] >= 0.010)
    crit3_B = bool(0.35 <= row_B["mean_ppr_050"] <= 0.65) and (row_B["degenerate_ppr_folds"] == 0)
    crit4_B = bool(row_B["roc_auc_degradation"] <= 0.001)
    pass_B = crit1_B and crit2_B and crit3_B and crit4_B

    crit1_C = bool(row_C["brier_reduction"] >= 0.005)
    crit2_C = bool(row_C["log_loss_reduction"] >= 0.010)
    crit3_C = bool(0.35 <= row_C["mean_ppr_050"] <= 0.65) and (row_C["degenerate_ppr_folds"] == 0)
    crit4_C = bool(row_C["roc_auc_degradation"] <= 0.001)
    pass_C = crit1_C and crit2_C and crit3_C and crit4_C

    df_sum["criterion_1_brier_pass"] = [True, crit1_B, crit1_C]
    df_sum["criterion_2_logloss_pass"] = [True, crit2_B, crit2_C]
    df_sum["criterion_3_ppr_pass"] = [True, crit3_B, crit3_C]
    df_sum["criterion_4_auc_pass"] = [True, crit4_B, crit4_C]
    df_sum["all_criteria_pass"] = [True, pass_B, pass_C]

    return df_sum


def _compute_asset_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute summary metrics grouped by asset."""
    records = []
    for asset_name, grp in df_fold.groupby("asset"):
        total = len(grp)
        records.append({
            "asset": asset_name,
            "total_folds": total,
            "brier_A": float(grp["brier_A"].mean()),
            "brier_B": float(grp["brier_B"].mean()),
            "brier_C": float(grp["brier_C"].mean()),
            "log_loss_A": float(grp["log_loss_A"].mean()),
            "log_loss_B": float(grp["log_loss_B"].mean()),
            "log_loss_C": float(grp["log_loss_C"].mean()),
            "ppr_A": float(grp["ppr_050_A"].mean()),
            "ppr_B": float(grp["ppr_050_B"].mean()),
            "ppr_C": float(grp["ppr_050_C"].mean()),
            "auc_A": float(grp["roc_auc_A"].mean()),
            "auc_B": float(grp["roc_auc_B"].mean()),
            "auc_C": float(grp["roc_auc_C"].mean()),
        })
    return pd.DataFrame(records)


def _compute_model_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute summary metrics grouped by model."""
    records = []
    for model_name, grp in df_fold.groupby("model"):
        total = len(grp)
        records.append({
            "model": model_name,
            "total_folds": total,
            "brier_A": float(grp["brier_A"].mean()),
            "brier_B": float(grp["brier_B"].mean()),
            "brier_C": float(grp["brier_C"].mean()),
            "log_loss_A": float(grp["log_loss_A"].mean()),
            "log_loss_B": float(grp["log_loss_B"].mean()),
            "log_loss_C": float(grp["log_loss_C"].mean()),
            "ppr_A": float(grp["ppr_050_A"].mean()),
            "ppr_B": float(grp["ppr_050_B"].mean()),
            "ppr_C": float(grp["ppr_050_C"].mean()),
            "auc_A": float(grp["roc_auc_A"].mean()),
            "auc_B": float(grp["roc_auc_B"].mean()),
            "auc_C": float(grp["roc_auc_C"].mean()),
        })
    return pd.DataFrame(records)


def _write_markdown_report(
    filepath: Path,
    df_fold: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_by_asset: pd.DataFrame,
    df_by_model: pd.DataFrame,
) -> None:
    lines = [
        "# Mission 14: Probability Calibration Experiment Report",
        "",
        f"**Total Outer Evaluations:** {len(df_fold)} across 5 assets, 3 models, and 5 outer expanding folds",
        "",
        "## Executive Summary",
        "",
        "This experiment evaluates Treatment A (Uncalibrated Baseline), Treatment B (Platt / Sigmoid Calibration), and Treatment C (Isotonic Calibration) using `SHORTLIST_16` across 5 liquid equity assets.",
        "",
        "## Aggregate Summary & Acceptance Criteria Audit",
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

    lines.append("## Per-Model Summary")
    lines.append("")
    if not df_by_model.empty:
        cols = list(df_by_model.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_by_model.iterrows():
            vals = [f"{r[c]:.4f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("## Per-Asset Summary")
    lines.append("")
    if not df_by_asset.empty:
        cols = list(df_by_asset.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_by_asset.iterrows():
            vals = [f"{r[c]:.4f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("## Complete Per-Fold Results")
    lines.append("")
    if not df_fold.empty:
        cols = [
            "asset", "model", "fold",
            "brier_A", "brier_B", "brier_C",
            "log_loss_A", "log_loss_B", "log_loss_C",
            "ppr_050_A", "ppr_050_B", "ppr_050_C",
            "roc_auc_A", "roc_auc_B", "roc_auc_C",
            "rank_corr_B", "rank_inversions_B",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_fold.iterrows():
            vals = [f"{r[c]:.4f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")

    filepath.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.perf_counter()
    logger.info("Executing Mission 14 Probability Calibration Experiment...")
    res = run_calibration_experiment()
    elapsed = time.perf_counter() - t0
    logger.info("Mission 14 Calibration Experiment completed in %.2f seconds.", elapsed)
