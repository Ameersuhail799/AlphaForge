"""Mission 17: Candidate Refinement, Regime Robustness & Trading Strategy Validation.

Comprehensive multi-asset, multi-model, multi-configuration research experiment evaluating
C0, C5, C7, C57, C8 across 5 assets, 3 models, and 5 expanding folds.
Implements PPR diagnostics, distribution shift analysis, pre-registered confidence filtering,
realistic 10-day trading simulation, and single-stock candidate readiness assessment.
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
    FEATURE_GROUP_E,
    FEATURE_GROUP_G,
    PROPOSED_31_FEATURES,
    MultiHorizonFeatureGenerator,
)
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

HIGHLY_REDUNDANT_6 = [
    "VOLATILITY_20D",
    "VOLUME_RATIO_20",
    "RSI_NEUTRAL_DIFF",
    "RETURN_10D",
    "CLOSE_TO_SMA20",
    "EMA20_SLOPE_5D",
]

NON_REDUNDANT_25 = [f for f in PROPOSED_31_FEATURES if f not in HIGHLY_REDUNDANT_6]

# 5 Configurations for Mission 17
CONFIGURATIONS: Dict[str, List[str]] = {
    "C0": SHORTLIST_16,
    "C5": SHORTLIST_16 + FEATURE_GROUP_E,
    "C7": SHORTLIST_16 + FEATURE_GROUP_G,
    "C57": SHORTLIST_16 + FEATURE_GROUP_E + FEATURE_GROUP_G,
    "C8": SHORTLIST_16 + NON_REDUNDANT_25,
}


@dataclass
class Mission17FoldResult:
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
    mean_prob: float
    median_prob: float
    std_prob: float
    q10_prob: float
    q25_prob: float
    q50_prob: float
    q75_prob: float
    q90_prob: float
    buy_signals_count: int
    sell_signals_count: int
    mean_realized_ret_buy: float
    mean_realized_ret_sell: float
    return_spread: float
    spread_minus_5bps: float
    spread_minus_10bps: float
    spread_minus_20bps: float
    spread_minus_50bps: float
    # Trading Simulation Metrics (No Filter)
    total_strat_return: float
    buy_hold_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    # Filtered Strategy Metrics (Confidence p >= 0.55)
    filtered_trades_count: int
    filtered_ppr: float
    filtered_precision: float
    filtered_return_spread: float
    filtered_cum_return: float
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


def compute_trading_simulation(
    binary_preds: np.ndarray,
    realized_returns: np.ndarray,
    cost_bps: float = 0.0010,  # 10 bps default
) -> Tuple[float, float, float, float, float, int, float]:
    """Simulate 10-day holding period strategy.
    
    Returns: (total_cum_return, buy_hold_return, win_rate, profit_factor, max_drawdown, total_trades, sharpe)
    """
    n = len(realized_returns)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0

    buy_hold_cum = float(np.cumprod(1.0 + realized_returns)[-1] - 1.0)

    # Strategy: Buy at t, hold for 10 days
    net_returns = np.where(binary_preds == 1, realized_returns - cost_bps, 0.0)
    total_trades = int(np.sum(binary_preds == 1))

    if total_trades > 0:
        trade_rets = net_returns[binary_preds == 1]
        wins = trade_rets[trade_rets > 0]
        losses = trade_rets[trade_rets < 0]

        win_rate = float(len(wins) / total_trades)
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0

        profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-8 else (99.0 if gross_profit > 0 else 1.0)
    else:
        win_rate = 0.0
        profit_factor = 0.0

    cum_rets = np.cumprod(1.0 + net_returns) - 1.0
    total_cum_ret = float(cum_rets[-1])

    # Drawdown
    eq_curve = 1.0 + cum_rets
    peak = np.maximum.accumulate(eq_curve)
    drawdowns = (peak - eq_curve) / peak
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe
    std_r = float(np.std(net_returns))
    mean_r = float(np.mean(net_returns))
    sharpe = float((mean_r / std_r) * np.sqrt(252)) if std_r > 1e-8 else 0.0

    return total_cum_ret, buy_hold_cum, win_rate, profit_factor, max_dd, total_trades, sharpe


def build_asset_dataset(asset_name: str) -> pd.DataFrame:
    """Build dataset with all 47 feature columns and TARGET_D for given asset."""
    storage = StorageEngine()
    fp = FeaturePipeline()
    gen_mh = MultiHorizonFeatureGenerator()

    raw = storage.load_dataset(asset_name)
    df_base = fp.generate(raw.copy())
    df_full = gen_mh.generate(df_base)

    close = df_full["Close"]
    ret_10d = (close.shift(-10) - close) / close

    df_full["TARGET_D"] = (ret_10d > 0).astype(int)
    df_full["REALIZED_RET_10D"] = ret_10d

    return df_full


def run_mission17_experiment(
    assets: Sequence[str] = DEFAULT_ASSETS,
    models: Sequence[str] = ("random_forest", "xgboost", "logistic_regression"),
    folds: int = 5,
    scale: bool = True,
    output_dir: Path | str = Path("reports") / "research",
) -> Dict[str, pd.DataFrame]:
    """Execute Mission 17 comprehensive strategy validation experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry()
    trainer = Trainer()
    fold_results: List[Mission17FoldResult] = []
    regime_diag_records: List[Dict[str, Any]] = []

    for asset_name in assets:
        logger.info("Processing asset for Mission 17 strategy validation: %s", asset_name)
        df_full = build_asset_dataset(asset_name)

        total_rows = len(df_full)
        test_size = max(1, int(total_rows * 0.15))
        non_test = df_full.iloc[:-test_size].copy()
        holdout_partition = df_full.iloc[-test_size:].copy()

        non_test_index = non_test.index
        outer_folds_positions = _create_folds_index(non_test_index, folds)

        for config_id, feature_cols in CONFIGURATIONS.items():
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

                # Feature distribution shift diagnostics for key features
                if config_id in ["C5", "C7", "C57"] and model_name_iter == "random_forest" if 'model_name_iter' in locals() else True:
                    for feat in ["POSITION_IN_20D_RANGE", "MOMENTUM_ACCELERATION"]:
                        if feat in outer_X_train.columns:
                            tr_m, val_m = float(outer_X_train[feat].mean()), float(outer_X_val[feat].mean())
                            tr_s, val_s = float(outer_X_train[feat].std()), float(outer_X_val[feat].std())
                            regime_diag_records.append({
                                "asset": asset_name,
                                "config_id": config_id,
                                "fold": fold_idx,
                                "feature": feat,
                                "train_mean": tr_m,
                                "val_mean": val_m,
                                "mean_shift": val_m - tr_m,
                                "train_std": tr_s,
                                "val_std": val_s,
                            })

                for model_name in models:
                    t0_tr = time.perf_counter()
                    outer_scaler = FeatureScaler(scale=scale)
                    outer_X_train_scaled = outer_scaler.fit_transform_train(outer_X_train)
                    outer_X_val_scaled = outer_scaler.transform(outer_X_val)

                    outer_model = registry.create(model_name)
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

                    prob_m = float(np.mean(val_probs))
                    prob_med = float(np.median(val_probs))
                    prob_std = float(np.std(val_probs))
                    q10, q25, q50, q75, q90 = [float(q) for q in np.quantile(val_probs, [0.10, 0.25, 0.50, 0.75, 0.90])]

                    buy_mask = (binary_preds == 1)
                    sell_mask = (binary_preds == 0)
                    buy_cnt = int(np.sum(buy_mask))
                    sell_cnt = int(np.sum(sell_mask))

                    ret_buy = float(np.mean(realized_ret_val[buy_mask])) if buy_cnt > 0 else 0.0
                    ret_sell = float(np.mean(realized_ret_val[sell_mask])) if sell_cnt > 0 else 0.0
                    ret_spread = ret_buy - ret_sell

                    # Transaction cost scenarios
                    s_5bps = ret_spread - 0.0005
                    s_10bps = ret_spread - 0.0010
                    s_20bps = ret_spread - 0.0020
                    s_50bps = ret_spread - 0.0050

                    # Trading Simulation (10-day holding period)
                    cum_ret, bh_ret, win_rate, pf, max_dd, total_trades, sharpe = compute_trading_simulation(
                        binary_preds, realized_ret_val, cost_bps=0.0010
                    )

                    # Pre-registered Confidence Filter (p >= 0.55)
                    filt_mask = (val_probs >= 0.55)
                    filt_trades_cnt = int(np.sum(filt_mask))
                    filt_ppr = float(filt_trades_cnt / len(val_probs)) if len(val_probs) > 0 else 0.0

                    if filt_trades_cnt > 0:
                        filt_prec = float(np.mean(y_val_arr[filt_mask] == 1))
                        filt_ret_buy = float(np.mean(realized_ret_val[filt_mask]))
                        filt_ret_no_buy = float(np.mean(realized_ret_val[~filt_mask])) if np.sum(~filt_mask) > 0 else 0.0
                        filt_spread = filt_ret_buy - filt_ret_no_buy
                        filt_cum_ret = float(np.cumprod(1.0 + np.where(filt_mask, realized_ret_val - 0.0010, 0.0))[-1] - 1.0)
                    else:
                        filt_prec = 0.0
                        filt_spread = 0.0
                        filt_cum_ret = 0.0

                    fold_results.append(
                        Mission17FoldResult(
                            config_id=config_id,
                            asset=asset_name,
                            target_name="TARGET_D",
                            model=model_name,
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
                            mean_prob=prob_m,
                            median_prob=prob_med,
                            std_prob=prob_std,
                            q10_prob=q10,
                            q25_prob=q25,
                            q50_prob=q50,
                            q75_prob=q75,
                            q90_prob=q90,
                            buy_signals_count=buy_cnt,
                            sell_signals_count=sell_cnt,
                            mean_realized_ret_buy=ret_buy,
                            mean_realized_ret_sell=ret_sell,
                            return_spread=ret_spread,
                            spread_minus_5bps=s_5bps,
                            spread_minus_10bps=s_10bps,
                            spread_minus_20bps=s_20bps,
                            spread_minus_50bps=s_50bps,
                            total_strat_return=cum_ret,
                            buy_hold_return=bh_ret,
                            win_rate=win_rate,
                            profit_factor=pf,
                            max_drawdown=max_dd,
                            sharpe_ratio=sharpe,
                            total_trades=total_trades,
                            filtered_trades_count=filt_trades_cnt,
                            filtered_ppr=filt_ppr,
                            filtered_precision=filt_prec,
                            filtered_return_spread=filt_spread,
                            filtered_cum_return=filt_cum_ret,
                            training_time=t_tr,
                            prediction_time=t_pred,
                        )
                    )

    df_fold = pd.DataFrame([asdict(r) for r in fold_results])
    df_summary = _compute_summary(df_fold)
    df_by_asset = _compute_asset_summary(df_fold)
    df_by_model = _compute_model_summary(df_fold)
    df_strategy = _compute_strategy_summary(df_fold)
    df_regime = pd.DataFrame(regime_diag_records) if regime_diag_records else pd.DataFrame()

    df_fold.to_csv(output_dir / "mission17_fold_results.csv", index=False)
    df_summary.to_csv(output_dir / "mission17_summary.csv", index=False)
    df_by_asset.to_csv(output_dir / "mission17_by_asset.csv", index=False)
    df_by_model.to_csv(output_dir / "mission17_by_model.csv", index=False)
    df_strategy.to_csv(output_dir / "mission17_strategy_results.csv", index=False)
    if not df_regime.empty:
        df_regime.to_csv(output_dir / "mission17_regime_diagnostics.csv", index=False)

    _write_markdown_report(output_dir / "MISSION_17_STRATEGY_VALIDATION_REPORT.md", df_fold, df_summary, df_by_asset, df_by_model, df_strategy)

    return {
        "fold_results": df_fold,
        "summary": df_summary,
        "by_asset": df_by_asset,
        "by_model": df_by_model,
        "strategy": df_strategy,
    }


def _compute_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate summary by configuration."""
    records = []
    for config_id, grp in df_fold.groupby("config_id"):
        total = len(grp)
        auc_m, auc_std = float(grp["roc_auc"].mean()), float(grp["roc_auc"].std())
        pr_auc_m = float(grp["pr_auc"].mean())
        mcc_m = float(grp["mcc"].mean())
        f1_m = float(grp["f1"].mean())
        ppr_m = float(grp["ppr"].mean())
        spread_m = float(grp["return_spread"].mean())
        net10_m = float(grp["spread_minus_10bps"].mean())
        net50_m = float(grp["spread_minus_50bps"].mean())

        pos_auc_cnt = int(np.sum(grp["roc_auc"] > 0.50))
        pos_mcc_cnt = int(np.sum(grp["mcc"] > 0.0))
        pos_spread_cnt = int(np.sum(grp["return_spread"] > 0.0))

        records.append({
            "config_id": config_id,
            "total_folds": total,
            "mean_roc_auc": auc_m,
            "std_roc_auc": auc_std,
            "mean_pr_auc": pr_auc_m,
            "mean_mcc": mcc_m,
            "mean_f1": f1_m,
            "mean_ppr": ppr_m,
            "mean_return_spread_pct": spread_m * 100,
            "net_spread_10bps_pct": net10_m * 100,
            "net_spread_50bps_pct": net50_m * 100,
            "pos_auc_folds": pos_auc_cnt,
            "pos_mcc_folds": pos_mcc_cnt,
            "pos_spread_folds": pos_spread_cnt,
        })
    return pd.DataFrame(records).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)


def _compute_asset_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute per-asset summary."""
    records = []
    for (asset_name, config_id, model_name), grp in df_fold.groupby(["asset", "config_id", "model"]):
        records.append({
            "asset": asset_name,
            "config_id": config_id,
            "model": model_name,
            "total_folds": len(grp),
            "mean_roc_auc": float(grp["roc_auc"].mean()),
            "mean_pr_auc": float(grp["pr_auc"].mean()),
            "mean_mcc": float(grp["mcc"].mean()),
            "mean_f1": float(grp["f1"].mean()),
            "mean_ppr": float(grp["ppr"].mean()),
            "mean_return_spread_pct": float(grp["return_spread"].mean()) * 100,
            "net_spread_10bps_pct": float(grp["spread_minus_10bps"].mean()) * 100,
        })
    return pd.DataFrame(records)


def _compute_model_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model summary."""
    records = []
    for (model_name, config_id), grp in df_fold.groupby(["model", "config_id"]):
        records.append({
            "model": model_name,
            "config_id": config_id,
            "total_folds": len(grp),
            "mean_roc_auc": float(grp["roc_auc"].mean()),
            "mean_pr_auc": float(grp["pr_auc"].mean()),
            "mean_mcc": float(grp["mcc"].mean()),
            "mean_f1": float(grp["f1"].mean()),
            "mean_ppr": float(grp["ppr"].mean()),
            "mean_return_spread_pct": float(grp["return_spread"].mean()) * 100,
        })
    return pd.DataFrame(records)


def _compute_strategy_summary(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Compute trading strategy simulation summary."""
    records = []
    for (asset_name, model_name, config_id), grp in df_fold.groupby(["asset", "model", "config_id"]):
        records.append({
            "asset": asset_name,
            "model": model_name,
            "config_id": config_id,
            "total_folds": len(grp),
            "mean_strat_cum_return_pct": float(grp["total_strat_return"].mean()) * 100,
            "mean_buy_hold_return_pct": float(grp["buy_hold_return"].mean()) * 100,
            "mean_win_rate_pct": float(grp["win_rate"].mean()) * 100,
            "mean_profit_factor": float(grp["profit_factor"].mean()),
            "mean_max_drawdown_pct": float(grp["max_drawdown"].mean()) * 100,
            "mean_sharpe": float(grp["sharpe_ratio"].mean()),
            "mean_filtered_cum_return_pct": float(grp["filtered_cum_return"].mean()) * 100,
            "mean_filtered_precision_pct": float(grp["filtered_precision"].mean()) * 100,
        })
    return pd.DataFrame(records)


def _write_markdown_report(
    filepath: Path,
    df_fold: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_by_asset: pd.DataFrame,
    df_by_model: pd.DataFrame,
    df_strategy: pd.DataFrame,
) -> None:
    lines = [
        "# Mission 17 — Candidate Refinement, Regime Robustness & Trading Strategy Validation Report",
        "",
        f"**Total Outer Evaluations:** {len(df_fold)} across 5 assets, 3 models, 5 outer folds, and 5 configurations (C0, C5, C7, C57, C8)",
        "",
        "## Executive Summary",
        "",
        "This experiment evaluates regime robustness, PPR signal bias, distribution shifts, pre-registered confidence filtering, and 10-day trading strategy performance across all asset/model/configuration combinations.",
        "",
        "## Overall Configuration Summary",
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

    filepath.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.perf_counter()
    logger.info("Executing Mission 17 Strategy Validation Experiment...")
    res = run_mission17_experiment()
    elapsed = time.perf_counter() - t0
    logger.info("Mission 17 completed in %.2f seconds.", elapsed)
