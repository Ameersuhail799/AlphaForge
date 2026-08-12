"""Mission 16 Step 2: Controlled Feature Family Experiment Runner.

Evaluates 9 pre-registered feature configurations (C0 through C8) on TCS + TARGET_D + Random Forest
across 5 chronological expanding folds to test feature family contributions against the C0 control benchmark.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.storage import StorageEngine
from src.dataset.scaler import FeatureScaler
from src.features.feature_pipeline import FeaturePipeline
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
from src.research.multi_horizon_feature_generator import (
    FEATURE_GROUP_A,
    FEATURE_GROUP_B,
    FEATURE_GROUP_C,
    FEATURE_GROUP_D,
    FEATURE_GROUP_E,
    FEATURE_GROUP_F,
    FEATURE_GROUP_G,
    PROPOSED_31_FEATURES,
    MultiHorizonFeatureGenerator,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

TCS_ASSET = "tcs_ns"

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

HIGHLY_REDUNDANT_6 = [
    "VOLATILITY_20D",
    "VOLUME_RATIO_20",
    "RSI_NEUTRAL_DIFF",
    "RETURN_10D",
    "CLOSE_TO_SMA20",
    "EMA20_SLOPE_5D",
]

NON_REDUNDANT_25 = [f for f in PROPOSED_31_FEATURES if f not in HIGHLY_REDUNDANT_6]

# Define 9 Configurations
CONFIGURATIONS: Dict[str, List[str]] = {
    "C0_Control": SHORTLIST_16,
    "C1_GroupA": SHORTLIST_16 + FEATURE_GROUP_A,
    "C2_GroupB": SHORTLIST_16 + FEATURE_GROUP_B,
    "C3_GroupC": SHORTLIST_16 + FEATURE_GROUP_C,
    "C4_GroupD": SHORTLIST_16 + FEATURE_GROUP_D,
    "C5_GroupE": SHORTLIST_16 + FEATURE_GROUP_E,
    "C6_GroupF": SHORTLIST_16 + FEATURE_GROUP_F,
    "C7_GroupG": SHORTLIST_16 + FEATURE_GROUP_G,
    "C8_Filtered_Combined": SHORTLIST_16 + NON_REDUNDANT_25,
}


@dataclass
class FeatureFoldResult:
    config_id: str
    asset: str
    target_name: str
    model: str
    fold: int
    feature_count: int
    train_samples: int
    validation_samples: int
    base_positive_rate: float
    roc_auc: float
    pr_auc: float
    mcc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    ppr: float
    mean_realized_ret_buy: float
    mean_realized_ret_sell: float
    return_spread: float
    spread_minus_5bps: float
    spread_minus_10bps: float
    spread_minus_20bps: float
    training_time: float
    prediction_time: float


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


def compute_pr_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Compute Precision-Recall Area Under Curve (PR-AUC) using trapezoidal integration."""
    y_arr = np.asarray(y_true, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)

    thresholds = np.sort(np.unique(probs_arr))[::-1]
    precisions = []
    recalls = []

    for t in thresholds:
        preds = (probs_arr >= t).astype(int)
        tp = np.sum((preds == 1) & (y_arr == 1))
        fp = np.sum((preds == 1) & (y_arr == 0))
        fn = np.sum((preds == 0) & (y_arr == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    recalls = np.array([0.0] + recalls + [1.0])
    base_rate = float(np.mean(y_arr))
    precisions = np.array([1.0] + precisions + [base_rate])

    order = np.argsort(recalls)
    pr_auc = float(np.trapezoid(precisions[order], recalls[order]))
    return np.clip(pr_auc, 0.0, 1.0)


def build_tcs_feature_dataset() -> pd.DataFrame:
    """Load TCS raw data and generate base pipeline + multi-horizon features."""
    storage = StorageEngine()
    fp = FeaturePipeline()
    gen_mh = MultiHorizonFeatureGenerator()

    raw = storage.load_dataset(TCS_ASSET)
    df_base = fp.generate(raw.copy())
    df_full = gen_mh.generate(df_base)

    close = df_full["Close"]
    ret_10d = (close.shift(-10) - close) / close

    df_full["TARGET_D"] = (ret_10d > 0).astype(int)
    df_full["REALIZED_RET_10D"] = ret_10d

    return df_full


def run_feature_family_experiment(
    folds: int = 5,
    scale: bool = True,
    output_dir: Path | str = Path("reports") / "research",
) -> Dict[str, pd.DataFrame]:
    """Execute controlled feature family experiment across configurations C0 to C8."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_full = build_tcs_feature_dataset()
    total_rows = len(df_full)
    test_size = max(1, int(total_rows * 0.15))

    non_test = df_full.iloc[:-test_size].copy()
    holdout_partition = df_full.iloc[-test_size:].copy()

    logger.info(
        "Mission 16 Experiment Setup: Total rows=%d, non-test=%d, isolated holdout=%d",
        total_rows,
        len(non_test),
        len(holdout_partition),
    )

    non_test_index = non_test.index
    outer_folds_positions = _create_folds_index(non_test_index, folds)

    registry = ModelRegistry()
    trainer = Trainer()
    fold_results: List[FeatureFoldResult] = []

    for config_id, feature_cols in CONFIGURATIONS.items():
        logger.info("Evaluating configuration %s (%d features)", config_id, len(feature_cols))

        valid_cols = [c for c in feature_cols if c in df_full.columns]
        required_cols = list(valid_cols) + ["TARGET_D", "REALIZED_RET_10D"]

        for fold_idx, (train_end_pos, val_end_pos) in enumerate(outer_folds_positions, start=1):
            train_end_idx = non_test_index[train_end_pos]
            val_start_idx = non_test_index[train_end_pos + 1]
            val_end_idx = non_test_index[val_end_pos]

            outer_train_raw = non_test.loc[:train_end_idx]
            outer_val_raw = non_test.loc[val_start_idx:val_end_idx]

            outer_train_df = outer_train_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()
            outer_val_df = outer_val_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()

            outer_X_train = outer_train_df[valid_cols].copy()
            outer_y_train = outer_train_df["TARGET_D"]
            outer_X_val = outer_val_df[valid_cols].copy()
            outer_y_val = outer_val_df["TARGET_D"]
            realized_ret_val = outer_val_df["REALIZED_RET_10D"].values

            base_pos_rate = float(np.mean(outer_y_train.values == 1))

            t0_tr = time.perf_counter()
            outer_scaler = FeatureScaler(scale=scale)
            outer_X_train_scaled = outer_scaler.fit_transform_train(outer_X_train)
            outer_X_val_scaled = outer_scaler.transform(outer_X_val)

            outer_model = registry.create("random_forest")
            train_bundle = type("TrainBundle", (), {"X_train": outer_X_train_scaled, "y_train": outer_y_train, "feature_names": valid_cols})()
            trainer.train(outer_model, train_bundle)
            t_tr = time.perf_counter() - t0_tr

            t0_pred = time.perf_counter()
            val_bundle = type("ValBundle", (), {"X_test": outer_X_val_scaled, "y_test": outer_y_val, "feature_names": valid_cols})()
            probs = outer_model.predict_proba(val_bundle)
            t_pred = time.perf_counter() - t0_pred

            val_probs = probs[:, 1] if (probs.ndim == 2 and probs.shape[1] == 2) else probs.ravel()
            binary_preds = (val_probs >= 0.50).astype(int)
            y_val_arr = outer_y_val.to_numpy()

            try:
                auc = float(roc_auc_score(y_val_arr, val_probs)) if len(np.unique(y_val_arr)) > 1 else 0.50
            except Exception:
                auc = 0.50

            pr_auc = compute_pr_auc(y_val_arr, val_probs)
            mcc = float(matthews_corrcoef(y_val_arr, binary_preds))
            f1 = float(f1_score(y_val_arr, binary_preds, zero_division=0))
            prec = float(precision_score(y_val_arr, binary_preds, zero_division=0))
            rec = float(recall_score(y_val_arr, binary_preds, zero_division=0))
            acc = float(accuracy_score(y_val_arr, binary_preds))
            ppr = float(np.mean(binary_preds))

            buy_mask = (binary_preds == 1)
            sell_mask = (binary_preds == 0)

            ret_buy = float(np.mean(realized_ret_val[buy_mask])) if np.sum(buy_mask) > 0 else 0.0
            ret_sell = float(np.mean(realized_ret_val[sell_mask])) if np.sum(sell_mask) > 0 else 0.0
            ret_spread = ret_buy - ret_sell

            # Transaction cost adjustments per trade
            s_5bps = ret_spread - 0.0005
            s_10bps = ret_spread - 0.0010
            s_20bps = ret_spread - 0.0020

            fold_results.append(
                FeatureFoldResult(
                    config_id=config_id,
                    asset=TCS_ASSET,
                    target_name="TARGET_D",
                    model="random_forest",
                    fold=fold_idx,
                    feature_count=len(valid_cols),
                    train_samples=len(outer_X_train_scaled),
                    validation_samples=len(outer_X_val_scaled),
                    base_positive_rate=base_pos_rate,
                    roc_auc=auc,
                    pr_auc=pr_auc,
                    mcc=mcc,
                    f1=f1,
                    precision=prec,
                    recall=rec,
                    accuracy=acc,
                    ppr=ppr,
                    mean_realized_ret_buy=ret_buy,
                    mean_realized_ret_sell=ret_sell,
                    return_spread=ret_spread,
                    spread_minus_5bps=s_5bps,
                    spread_minus_10bps=s_10bps,
                    spread_minus_20bps=s_20bps,
                    training_time=t_tr,
                    prediction_time=t_pred,
                )
            )

    df_fold = pd.DataFrame([asdict(r) for r in fold_results])
    df_summary, df_stability = _compute_summary_and_stability(df_fold)

    df_fold.to_csv(output_dir / "mission16_feature_fold_results.csv", index=False)
    df_fold.to_csv(output_dir / "mission16_feature_experiment.csv", index=False)
    df_summary.to_csv(output_dir / "mission16_feature_summary.csv", index=False)
    df_stability.to_csv(output_dir / "mission16_feature_stability.csv", index=False)

    _write_markdown_report(output_dir / "MISSION_16_FEATURE_EXPERIMENT_REPORT.md", df_fold, df_summary, df_stability)

    return {
        "fold_results": df_fold,
        "summary": df_summary,
        "stability": df_stability,
    }


def _compute_summary_and_stability(df_fold: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute summary statistics, stability metrics, transaction cost diagnostics, and winner audit."""
    summary_records = []
    stability_records = []

    # Get Control C0 metrics for direct comparison
    c0_df = df_fold[df_fold["config_id"] == "C0_Control"]
    c0_mean_auc = float(c0_df["roc_auc"].mean())
    c0_mean_mcc = float(c0_df["mcc"].mean())
    c0_mean_spread = float(c0_df["return_spread"].mean())

    for config_id, grp in df_fold.groupby("config_id"):
        total_folds = len(grp)
        feat_cnt = grp["feature_count"].iloc[0]

        auc_vals = grp["roc_auc"].values
        pr_auc_vals = grp["pr_auc"].values
        mcc_vals = grp["mcc"].values
        f1_vals = grp["f1"].values
        spread_vals = grp["return_spread"].values
        ppr_vals = grp["ppr"].values

        auc_mean, auc_median, auc_std, auc_min, auc_max = float(np.mean(auc_vals)), float(np.median(auc_vals)), float(np.std(auc_vals)), float(np.min(auc_vals)), float(np.max(auc_vals))
        pr_auc_mean = float(np.mean(pr_auc_vals))
        mcc_mean, mcc_std = float(np.mean(mcc_vals)), float(np.std(mcc_vals))
        f1_mean = float(np.mean(f1_vals))
        spread_mean, spread_std = float(np.mean(spread_vals)), float(np.std(spread_vals))

        s_5bps_mean = float(np.mean(grp["spread_minus_5bps"].values))
        s_10bps_mean = float(np.mean(grp["spread_minus_10bps"].values))
        s_20bps_mean = float(np.mean(grp["spread_minus_20bps"].values))

        pos_auc_folds = int(np.sum(auc_vals > 0.50))
        pos_mcc_folds = int(np.sum(mcc_vals > 0.0))
        pos_spread_folds = int(np.sum(spread_vals > 0.0))

        degen_ppr_folds = int(np.sum((ppr_vals > 0.80) | (ppr_vals < 0.20)))

        # Audit against the 7 Winner Rules:
        rule1_auc = (auc_mean >= c0_mean_auc) or (auc_mean >= 0.550 and spread_mean > c0_mean_spread)
        rule2_mcc = (mcc_mean > c0_mean_mcc)
        rule3_spread = (spread_mean > c0_mean_spread)
        rule4_pos_mcc = (pos_mcc_folds >= 4)
        rule5_pos_spread = (pos_spread_folds >= 4)
        rule6_no_degen = (degen_ppr_folds == 0)
        rule7_no_outlier = (auc_min >= 0.500)

        is_promising = rule1_auc and rule2_mcc and rule3_spread and rule4_pos_mcc and rule5_pos_spread and rule6_no_degen and rule7_no_outlier

        summary_records.append({
            "config_id": config_id,
            "feature_count": feat_cnt,
            "total_folds": total_folds,
            "mean_roc_auc": auc_mean,
            "diff_auc_vs_control": auc_mean - c0_mean_auc,
            "median_roc_auc": auc_median,
            "std_roc_auc": auc_std,
            "min_roc_auc": auc_min,
            "max_roc_auc": auc_max,
            "mean_pr_auc": pr_auc_mean,
            "mean_mcc": mcc_mean,
            "diff_mcc_vs_control": mcc_mean - c0_mean_mcc,
            "mean_f1": f1_mean,
            "mean_ppr": float(np.mean(ppr_vals)),
            "mean_return_spread_pct": spread_mean * 100,
            "diff_spread_vs_control_pct": (spread_mean - c0_mean_spread) * 100,
            "spread_minus_5bps_pct": s_5bps_mean * 100,
            "spread_minus_10bps_pct": s_10bps_mean * 100,
            "spread_minus_20bps_pct": s_20bps_mean * 100,
            "pos_auc_folds": pos_auc_folds,
            "pos_mcc_folds": pos_mcc_folds,
            "pos_spread_folds": pos_spread_folds,
            "degenerate_ppr_folds": degen_ppr_folds,
            "is_promising": is_promising,
        })

        stability_records.append({
            "config_id": config_id,
            "pos_auc_rate": pos_auc_folds / total_folds,
            "pos_mcc_rate": pos_mcc_folds / total_folds,
            "pos_spread_rate": pos_spread_folds / total_folds,
            "worst_fold_mcc": float(np.min(mcc_vals)),
            "worst_fold_spread_pct": float(np.min(spread_vals)) * 100,
            "fold_std_roc_auc": auc_std,
            "fold_std_mcc": mcc_std,
            "fold_std_spread_pct": spread_std * 100,
        })

    df_summary = pd.DataFrame(summary_records).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)
    df_stability = pd.DataFrame(stability_records)

    return df_summary, df_stability


def _write_markdown_report(
    filepath: Path,
    df_fold: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_stability: pd.DataFrame,
) -> None:
    lines = [
        "# Mission 16 — Step 2: Controlled Feature Family Experiment Report",
        "",
        "## Executive Summary",
        "",
        "This experiment evaluates 9 pre-registered feature configurations (C0 through C8) on `tcs_ns` + `TARGET_D` (10-Day Direction) + `random_forest` across 5 chronological expanding-window folds.",
        "",
        "## Configuration Summary Table",
        "",
    ]

    if not df_summary.empty:
        cols = ["config_id", "feature_count", "mean_roc_auc", "diff_auc_vs_control", "mean_pr_auc", "mean_mcc", "diff_mcc_vs_control", "mean_f1", "mean_ppr", "mean_return_spread_pct", "spread_minus_10bps_pct", "pos_mcc_folds", "is_promising"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_summary.iterrows():
            vals = [f"{r[c]:.4f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("## Fold-by-Fold Matrix Across Configurations")
    lines.append("")
    if not df_fold.empty:
        cols = ["config_id", "fold", "feature_count", "roc_auc", "pr_auc", "mcc", "f1", "ppr", "return_spread", "spread_minus_10bps"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_fold.iterrows():
            vals = [f"{r[c]:.4f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")

    filepath.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.perf_counter()
    logger.info("Executing Mission 16 Step 2 Controlled Feature Family Experiment...")
    res = run_feature_family_experiment()
    elapsed = time.perf_counter() - t0
    logger.info("Mission 16 Controlled Feature Family Experiment completed in %.2f seconds.", elapsed)
