"""Mission 15 Step 2: Target Horizon Screening Experiment Runner.

Evaluates Targets A, B, C, D, E, F across 5 assets, 3 models, and 5 expanding outer folds
holding feature set constant at SHORTLIST_16 to isolate target reformulation effects.
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
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.storage import StorageEngine
from src.dataset.scaler import FeatureScaler
from src.dataset.target import TargetBuilder
from src.features.feature_pipeline import FeaturePipeline
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
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
class TargetScreeningFoldResult:
    asset: str
    model: str
    fold: int
    target_name: str
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

    # Append boundary points (Recall=0, Prec=1) and (Recall=1, Prec=base_rate)
    recalls = np.array([0.0] + recalls + [1.0])
    base_rate = float(np.mean(y_arr))
    precisions = np.array([1.0] + precisions + [base_rate])

    # Sort by recall for trapezoidal integration
    order = np.argsort(recalls)
    pr_auc = float(np.trapezoid(precisions[order], recalls[order]))
    return np.clip(pr_auc, 0.0, 1.0)


def build_candidate_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Build candidate targets A through F on market feature dataframe."""
    df = df.copy()
    close = df["Close"]
    atr = df["ATR_14"] if "ATR_14" in df.columns else (df["High"] - df["Low"]).rolling(14).mean()

    # Forward returns
    ret_1d = (close.shift(-1) - close) / close
    ret_3d = (close.shift(-3) - close) / close
    ret_5d = (close.shift(-5) - close) / close
    ret_10d = (close.shift(-10) - close) / close

    atr_pct = atr / close

    # TARGET_A: 1D Direction
    df["TARGET_A"] = (ret_1d > 0).astype(int)

    # TARGET_B: 3D Direction
    df["TARGET_B"] = (ret_3d > 0).astype(int)

    # TARGET_C: 5D Direction
    df["TARGET_C"] = (ret_5d > 0).astype(int)

    # TARGET_D: 10D Direction
    df["TARGET_D"] = (ret_10d > 0).astype(int)

    # TARGET_E: 5D Volatility-Normalized Return Direction
    vol_norm_ret_5d = ret_5d / (atr_pct + 1e-8)
    df["TARGET_E"] = (vol_norm_ret_5d > 0).astype(int)

    # TARGET_F: 5D Neutral-Band Target (0 = SELL, 1 = NO_TRADE, 2 = BUY)
    upper_thresh = 0.75 * atr_pct
    lower_thresh = -0.75 * atr_pct

    target_f = np.ones(len(df), dtype=int)  # default 1: NO_TRADE
    target_f[ret_5d > upper_thresh] = 2      # 2: BUY
    target_f[ret_5d < lower_thresh] = 0      # 0: SELL
    df["TARGET_F"] = target_f

    # Store realized returns for economic diagnostics
    df["REALIZED_RET_1D"] = ret_1d
    df["REALIZED_RET_5D"] = ret_5d

    return df


def run_target_screening_experiment(
    assets: Sequence[str] = DEFAULT_ASSETS,
    folds: int = 5,
    scale: bool = True,
    output_dir: Path | str = Path("reports") / "research",
) -> Dict[str, pd.DataFrame]:
    """Execute target horizon screening experiment across assets, models, targets, and expanding folds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = StorageEngine()
    fp = FeaturePipeline()
    tb = TargetBuilder()
    registry = ModelRegistry()
    trainer = Trainer()

    models = registry.list_models()
    target_names = ["TARGET_A", "TARGET_B", "TARGET_C", "TARGET_D", "TARGET_E", "TARGET_F"]
    fold_results: List[TargetScreeningFoldResult] = []

    for asset_name in assets:
        logger.info("Processing asset for Mission 15 target screening: %s", asset_name)
        if not storage.dataset_exists(asset_name):
            logger.warning("Dataset %s not found in storage. Skipping.", asset_name)
            continue

        raw = storage.load_dataset(asset_name)
        df_features = fp.generate(raw.copy())
        df_with_targets = build_candidate_targets(df_features)

        total_rows = len(df_with_targets)
        test_size = max(1, int(total_rows * 0.15))
        non_test = df_with_targets.iloc[:-test_size].copy()
        holdout_partition = df_with_targets.iloc[-test_size:].copy()

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

            outer_train_df_raw = non_test.loc[:train_end_idx]
            outer_val_df_raw = non_test.loc[val_start_idx:val_end_idx]

            for target_name in target_names:
                required = list(feature_cols) + [target_name, "REALIZED_RET_5D"]
                outer_train_df = outer_train_df_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
                outer_val_df = outer_val_df_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()

                if outer_train_df.empty or outer_val_df.empty:
                    continue

                outer_X_train = outer_train_df[feature_cols].copy()
                outer_y_train = outer_train_df[target_name]
                outer_X_val = outer_val_df[feature_cols].copy()
                outer_y_val = outer_val_df[target_name]

                realized_ret_val = outer_val_df["REALIZED_RET_5D"].values

                base_pos_rate = float(np.mean(outer_y_train.values == (2 if target_name == "TARGET_F" else 1)))

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
                    preds = outer_model.predict(outer_val_bundle)
                    t_pred = time.perf_counter() - t0_pred

                    y_val_arr = outer_y_val.to_numpy()

                    if target_name == "TARGET_F":
                        # Tri-class evaluation (0=SELL, 1=NO_TRADE, 2=BUY)
                        macro_f1 = float(f1_score(y_val_arr, preds, average="macro", zero_division=0))
                        bal_acc = float(balanced_accuracy_score(y_val_arr, preds))
                        mcc = float(matthews_corrcoef(y_val_arr, preds))
                        acc = float(accuracy_score(y_val_arr, preds))
                        prec = float(precision_score(y_val_arr, preds, average="macro", zero_division=0))
                        rec = float(recall_score(y_val_arr, preds, average="macro", zero_division=0))

                        try:
                            if probs.ndim == 2 and probs.shape[1] == 3:
                                auc = float(roc_auc_score(y_val_arr, probs, multi_class="ovr"))
                            else:
                                auc = 0.50
                        except Exception:
                            auc = 0.50

                        pr_auc = float(compute_pr_auc((y_val_arr == 2).astype(int), probs[:, 2] if (probs.ndim == 2 and probs.shape[1] == 3) else (preds == 2).astype(int)))
                        ppr = float(np.mean(preds == 2))

                        buy_mask = (preds == 2)
                        sell_mask = (preds == 0)
                        ret_buy = float(np.mean(realized_ret_val[buy_mask])) if np.sum(buy_mask) > 0 else 0.0
                        ret_sell = float(np.mean(realized_ret_val[sell_mask])) if np.sum(sell_mask) > 0 else 0.0

                    else:
                        # Binary evaluation (Targets A, B, C, D, E)
                        val_probs = probs[:, 1] if (probs.ndim == 2 and probs.shape[1] == 2) else probs.ravel()
                        binary_preds = (val_probs >= 0.50).astype(int)

                        try:
                            if len(np.unique(y_val_arr)) > 1:
                                auc = float(roc_auc_score(y_val_arr, val_probs))
                            else:
                                auc = 0.50
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

                    fold_results.append(
                        TargetScreeningFoldResult(
                            asset=asset_name,
                            model=model_name,
                            fold=fold_idx,
                            target_name=target_name,
                            train_samples=len(outer_X_train_scaled),
                            validation_samples=len(outer_X_val_scaled),
                            base_positive_rate=base_pos_rate,
                            roc_auc=auc,
                            pr_auc=pr_auc,
                            mcc=mcc,
                            f1=f1 if target_name != "TARGET_F" else macro_f1,
                            precision=prec,
                            recall=rec,
                            accuracy=acc if target_name != "TARGET_F" else bal_acc,
                            ppr=ppr,
                            mean_realized_ret_buy=ret_buy,
                            mean_realized_ret_sell=ret_sell,
                            training_time=t_tr,
                            prediction_time=t_pred,
                        )
                    )

    df_fold = pd.DataFrame([asdict(r) for r in fold_results])
    df_summary = _compute_summary(df_fold)
    df_by_asset = _compute_asset_summary(df_fold)
    df_by_model = _compute_model_summary(df_fold)

    df_fold.to_csv(output_dir / "mission15_target_screening.csv", index=False)
    df_fold.to_csv(output_dir / "mission15_target_fold_results.csv", index=False)
    df_summary.to_csv(output_dir / "mission15_target_summary.csv", index=False)
    df_by_asset.to_csv(output_dir / "mission15_target_by_asset.csv", index=False)
    df_by_model.to_csv(output_dir / "mission15_target_by_model.csv", index=False)

    _write_markdown_report(output_dir / "MISSION_15_TARGET_SCREENING_REPORT.md", df_fold, df_summary, df_by_asset, df_by_model)

    return {
        "fold_results": df_fold,
        "summary": df_summary,
        "by_asset": df_by_asset,
        "by_model": df_by_model,
    }


def _compute_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate summary metrics by target candidate."""
    records = []
    for target_name, grp in df_fold.groupby("target_name"):
        total = len(grp)
        auc_m = float(grp["roc_auc"].mean())
        pr_auc_m = float(grp["pr_auc"].mean())
        mcc_m = float(grp["mcc"].mean())
        f1_m = float(grp["f1"].mean())
        prec_m = float(grp["precision"].mean())
        rec_m = float(grp["recall"].mean())
        acc_m = float(grp["accuracy"].mean())
        ppr_m = float(grp["ppr"].mean())
        ret_buy_m = float(grp["mean_realized_ret_buy"].mean())
        ret_sell_m = float(grp["mean_realized_ret_sell"].mean())

        records.append({
            "target_name": target_name,
            "total_folds": total,
            "mean_roc_auc": auc_m,
            "mean_pr_auc": pr_auc_m,
            "mean_mcc": mcc_m,
            "mean_f1": f1_m,
            "mean_precision": prec_m,
            "mean_recall": rec_m,
            "mean_accuracy": acc_m,
            "mean_ppr": ppr_m,
            "mean_realized_ret_buy_pct": ret_buy_m * 100,
            "mean_realized_ret_sell_pct": ret_sell_m * 100,
            "return_spread_pct": (ret_buy_m - ret_sell_m) * 100,
        })
    return pd.DataFrame(records).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)


def _compute_asset_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute per-asset summary metrics across target candidates."""
    records = []
    for (asset_name, target_name), grp in df_fold.groupby(["asset", "target_name"]):
        records.append({
            "asset": asset_name,
            "target_name": target_name,
            "total_folds": len(grp),
            "mean_roc_auc": float(grp["roc_auc"].mean()),
            "mean_pr_auc": float(grp["pr_auc"].mean()),
            "mean_mcc": float(grp["mcc"].mean()),
            "mean_f1": float(grp["f1"].mean()),
            "mean_ppr": float(grp["ppr"].mean()),
            "return_spread_pct": (float(grp["mean_realized_ret_buy"].mean()) - float(grp["mean_realized_ret_sell"].mean())) * 100,
        })
    return pd.DataFrame(records)


def _compute_model_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model summary metrics across target candidates."""
    records = []
    for (model_name, target_name), grp in df_fold.groupby(["model", "target_name"]):
        records.append({
            "model": model_name,
            "target_name": target_name,
            "total_folds": len(grp),
            "mean_roc_auc": float(grp["roc_auc"].mean()),
            "mean_pr_auc": float(grp["pr_auc"].mean()),
            "mean_mcc": float(grp["mcc"].mean()),
            "mean_f1": float(grp["f1"].mean()),
            "mean_ppr": float(grp["ppr"].mean()),
            "return_spread_pct": (float(grp["mean_realized_ret_buy"].mean()) - float(grp["mean_realized_ret_sell"].mean())) * 100,
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
        "# Mission 15 — Step 2: Target Horizon Screening Report",
        "",
        f"**Total Outer Evaluations:** {len(df_fold)} across 5 assets, 3 models, 5 outer folds, and 6 candidate targets (A–F)",
        "",
        "## Executive Summary",
        "",
        "This experiment evaluates candidate target horizons (TARGET_A through TARGET_F) holding feature set constant at `SHORTLIST_16` to isolate target reformulation effects.",
        "",
        "## Target Summary Table",
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

    filepath.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.perf_counter()
    logger.info("Executing Mission 15 Step 2 Target Horizon Screening Experiment...")
    res = run_target_screening_experiment()
    elapsed = time.perf_counter() - t0
    logger.info("Mission 15 Target Horizon Screening completed in %.2f seconds.", elapsed)
