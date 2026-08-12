"""Mission 15 Step 3: TCS Candidate Tournament & Deep Validation.

Evaluates exactly four TCS candidate combinations across 5 chronological expanding folds
to analyze temporal stability, worst-fold performance, return spread, and drawdown.
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
from src.dataset.target import TargetBuilder
from src.features.feature_pipeline import FeaturePipeline
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
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


@dataclass
class TCSCandidateSpec:
    candidate_id: str
    asset: str
    target_name: str
    target_horizon: int
    model_name: str


TOURNAMENT_CANDIDATES = [
    TCSCandidateSpec("Candidate_A", TCS_ASSET, "TARGET_B", 3, "random_forest"),
    TCSCandidateSpec("Candidate_B", TCS_ASSET, "TARGET_B", 3, "xgboost"),
    TCSCandidateSpec("Candidate_C", TCS_ASSET, "TARGET_D", 10, "random_forest"),
    TCSCandidateSpec("Candidate_D", TCS_ASSET, "TARGET_D", 10, "xgboost"),
]


@dataclass
class TCSFoldEvaluationResult:
    candidate_id: str
    asset: str
    target_name: str
    model: str
    fold: int
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
    buy_signals_count: int
    sell_signals_count: int
    mean_realized_ret_buy: float
    mean_realized_ret_sell: float
    return_spread: float
    cum_strategy_return: float
    max_drawdown: float
    sharpe_ratio: float
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


def compute_strategy_performance(signals: np.ndarray, realized_returns: np.ndarray) -> Tuple[float, float, float]:
    """Compute simple long/flat strategy cumulative return, max drawdown, and Sharpe ratio.
    
    signals: binary 1 (BUY) or 0 (FLAT/SELL)
    realized_returns: per-period returns
    """
    strat_returns = np.where(signals == 1, realized_returns, 0.0)
    cum_returns = np.cumprod(1.0 + strat_returns) - 1.0
    total_cum_ret = float(cum_returns[-1]) if len(cum_returns) > 0 else 0.0

    # Max Drawdown
    equity_curve = 1.0 + cum_returns
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / peak
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe Ratio
    std_ret = float(np.std(strat_returns))
    mean_ret = float(np.mean(strat_returns))
    sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret > 1e-8 else 0.0

    return total_cum_ret, max_dd, sharpe


def build_tcs_tournament_df() -> pd.DataFrame:
    """Load TCS raw dataset and build candidate target labels."""
    storage = StorageEngine()
    fp = FeaturePipeline()

    raw = storage.load_dataset(TCS_ASSET)
    df_features = fp.generate(raw.copy())
    close = df_features["Close"]

    ret_3d = (close.shift(-3) - close) / close
    ret_10d = (close.shift(-10) - close) / close

    df_features["TARGET_B"] = (ret_3d > 0).astype(int)
    df_features["TARGET_D"] = (ret_10d > 0).astype(int)
    df_features["REALIZED_RET_3D"] = ret_3d
    df_features["REALIZED_RET_10D"] = ret_10d

    return df_features


def run_tcs_candidate_tournament(
    folds: int = 5,
    scale: bool = True,
    output_dir: Path | str = Path("reports") / "research",
) -> Dict[str, pd.DataFrame]:
    """Execute the TCS Candidate Tournament across 5 chronological expanding folds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_full = build_tcs_tournament_df()
    total_rows = len(df_full)
    test_size = max(1, int(total_rows * 0.15))

    non_test = df_full.iloc[:-test_size].copy()
    holdout_partition = df_full.iloc[-test_size:].copy()

    logger.info(
        "TCS Tournament Setup: Total rows=%d, non-test=%d, isolated holdout=%d",
        total_rows,
        len(non_test),
        len(holdout_partition),
    )

    non_test_index = non_test.index
    outer_folds_positions = _create_folds_index(non_test_index, folds)
    feature_cols = [c for c in SHORTLIST_16 if c in df_full.columns]

    registry = ModelRegistry()
    trainer = Trainer()
    fold_results: List[TCSFoldEvaluationResult] = []

    for cand in TOURNAMENT_CANDIDATES:
        logger.info("Evaluating %s: Target=%s, Model=%s", cand.candidate_id, cand.target_name, cand.model_name)

        realized_col = f"REALIZED_RET_{cand.target_horizon}D"
        required_cols = list(feature_cols) + [cand.target_name, realized_col]

        for fold_idx, (train_end_pos, val_end_pos) in enumerate(outer_folds_positions, start=1):
            train_end_idx = non_test_index[train_end_pos]
            val_start_idx = non_test_index[train_end_pos + 1]
            val_end_idx = non_test_index[val_end_pos]

            outer_train_raw = non_test.loc[:train_end_idx]
            outer_val_raw = non_test.loc[val_start_idx:val_end_idx]

            outer_train_df = outer_train_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()
            outer_val_df = outer_val_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()

            outer_X_train = outer_train_df[feature_cols].copy()
            outer_y_train = outer_train_df[cand.target_name]
            outer_X_val = outer_val_df[feature_cols].copy()
            outer_y_val = outer_val_df[cand.target_name]
            realized_ret_val = outer_val_df[realized_col].values

            base_pos_rate = float(np.mean(outer_y_train.values == 1))

            t0_tr = time.perf_counter()
            outer_scaler = FeatureScaler(scale=scale)
            outer_X_train_scaled = outer_scaler.fit_transform_train(outer_X_train)
            outer_X_val_scaled = outer_scaler.transform(outer_X_val)

            outer_model = registry.create(cand.model_name)
            train_bundle = type("TrainBundle", (), {"X_train": outer_X_train_scaled, "y_train": outer_y_train, "feature_names": feature_cols})()
            trainer.train(outer_model, train_bundle)
            t_tr = time.perf_counter() - t0_tr

            t0_pred = time.perf_counter()
            val_bundle = type("ValBundle", (), {"X_test": outer_X_val_scaled, "y_test": outer_y_val, "feature_names": feature_cols})()
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

            buy_cnt = int(np.sum(buy_mask))
            sell_cnt = int(np.sum(sell_mask))

            ret_buy = float(np.mean(realized_ret_val[buy_mask])) if buy_cnt > 0 else 0.0
            ret_sell = float(np.mean(realized_ret_val[sell_mask])) if sell_cnt > 0 else 0.0
            ret_spread = ret_buy - ret_sell

            cum_ret, max_dd, sharpe = compute_strategy_performance(binary_preds, realized_ret_val)

            fold_results.append(
                TCSFoldEvaluationResult(
                    candidate_id=cand.candidate_id,
                    asset=cand.asset,
                    target_name=cand.target_name,
                    model=cand.model_name,
                    fold=fold_idx,
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
                    buy_signals_count=buy_cnt,
                    sell_signals_count=sell_cnt,
                    mean_realized_ret_buy=ret_buy,
                    mean_realized_ret_sell=ret_sell,
                    return_spread=ret_spread,
                    cum_strategy_return=cum_ret,
                    max_drawdown=max_dd,
                    sharpe_ratio=sharpe,
                    training_time=t_tr,
                    prediction_time=t_pred,
                )
            )

    df_fold = pd.DataFrame([asdict(r) for r in fold_results])
    df_summary, df_ranking = _compute_candidate_summary_and_ranking(df_fold)

    df_fold.to_csv(output_dir / "tcs_candidate_tournament_fold_results.csv", index=False)
    df_summary.to_csv(output_dir / "tcs_candidate_tournament_summary.csv", index=False)
    df_ranking.to_csv(output_dir / "tcs_candidate_tournament_ranking.csv", index=False)

    _write_markdown_report(output_dir / "MISSION_15_TCS_CANDIDATE_TOURNAMENT_REPORT.md", df_fold, df_summary, df_ranking)

    return {
        "fold_results": df_fold,
        "summary": df_summary,
        "ranking": df_ranking,
    }


def compute_stability_score(pos_auc_rate: float, pos_mcc_rate: float, pos_spread_rate: float, std_auc: float) -> float:
    """Compute research-only Stability Score.
    
    Formula: (0.35 * pos_auc_rate + 0.35 * pos_mcc_rate + 0.30 * pos_spread_rate - 0.50 * std_auc) * 100
    """
    raw_score = (0.35 * pos_auc_rate) + (0.35 * pos_mcc_rate) + (0.30 * pos_spread_rate) - (0.50 * std_auc)
    return float(np.clip(raw_score * 100.0, 0.0, 100.0))


def _compute_candidate_summary_and_ranking(df_fold: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute stability statistics, worst/best fold analysis, stability score, and ranking."""
    summary_records = []

    for cand_id, grp in df_fold.groupby("candidate_id"):
        n_folds = len(grp)
        target_name = grp["target_name"].iloc[0]
        model_name = grp["model"].iloc[0]

        auc_vals = grp["roc_auc"].values
        pr_auc_vals = grp["pr_auc"].values
        mcc_vals = grp["mcc"].values
        f1_vals = grp["f1"].values
        spread_vals = grp["return_spread"].values

        auc_mean, auc_median, auc_std, auc_min, auc_max = float(np.mean(auc_vals)), float(np.median(auc_vals)), float(np.std(auc_vals)), float(np.min(auc_vals)), float(np.max(auc_vals))
        pr_auc_mean, pr_auc_std = float(np.mean(pr_auc_vals)), float(np.std(pr_auc_vals))
        mcc_mean, mcc_std = float(np.mean(mcc_vals)), float(np.std(mcc_vals))
        f1_mean, f1_std = float(np.mean(f1_vals)), float(np.std(f1_vals))
        spread_mean, spread_std = float(np.mean(spread_vals)), float(np.std(spread_vals))

        pos_auc_folds = int(np.sum(auc_vals > 0.50))
        pos_mcc_folds = int(np.sum(mcc_vals > 0.0))
        pos_spread_folds = int(np.sum(spread_vals > 0.0))

        pos_auc_rate = pos_auc_folds / n_folds
        pos_mcc_rate = pos_mcc_folds / n_folds
        pos_spread_rate = pos_spread_folds / n_folds

        stability_score = compute_stability_score(pos_auc_rate, pos_mcc_rate, pos_spread_rate, auc_std)

        worst_fold_idx = int(grp.loc[grp["roc_auc"].idxmin(), "fold"])
        best_fold_idx = int(grp.loc[grp["roc_auc"].idxmax(), "fold"])

        summary_records.append({
            "candidate_id": cand_id,
            "target_name": target_name,
            "model": model_name,
            "total_folds": n_folds,
            "mean_roc_auc": auc_mean,
            "median_roc_auc": auc_median,
            "std_roc_auc": auc_std,
            "min_roc_auc": auc_min,
            "max_roc_auc": auc_max,
            "mean_pr_auc": pr_auc_mean,
            "std_pr_auc": pr_auc_std,
            "mean_mcc": mcc_mean,
            "std_mcc": mcc_std,
            "mean_f1": f1_mean,
            "std_f1": f1_std,
            "mean_return_spread_pct": spread_mean * 100,
            "std_return_spread_pct": spread_std * 100,
            "pos_auc_folds": pos_auc_folds,
            "pos_mcc_folds": pos_mcc_folds,
            "pos_spread_folds": pos_spread_folds,
            "stability_score": stability_score,
            "worst_fold": worst_fold_idx,
            "worst_fold_auc": auc_min,
            "best_fold": best_fold_idx,
            "best_fold_auc": auc_max,
        })

    df_summary = pd.DataFrame(summary_records)

    # Ranking logic:
    # 1. Stability Score (Consistency across folds)
    # 2. Positive MCC Folds
    # 3. Mean Return Spread
    # 4. Mean ROC-AUC
    # 5. Mean PR-AUC
    df_ranking = df_summary.sort_values(
        by=["stability_score", "pos_mcc_folds", "mean_return_spread_pct", "mean_roc_auc", "mean_pr_auc"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    # Add Verdict
    verdicts = []
    for _, r in df_ranking.iterrows():
        if r["stability_score"] >= 75.0 and r["pos_auc_folds"] >= 4 and r["pos_mcc_folds"] >= 4 and r["pos_spread_folds"] >= 4:
            verdicts.append("🟢 STRONG CANDIDATE")
        elif r["pos_auc_folds"] >= 3 or r["pos_mcc_folds"] >= 3:
            verdicts.append("🟡 PROMISING BUT UNSTABLE")
        else:
            verdicts.append("🔴 REJECT")
    df_ranking["verdict"] = verdicts

    return df_summary, df_ranking


def _write_markdown_report(
    filepath: Path,
    df_fold: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_ranking: pd.DataFrame,
) -> None:
    lines = [
        "# Mission 15 — Step 3: TCS Candidate Tournament & Deep Validation Report",
        "",
        "## Executive Summary",
        "",
        "This experiment deep-validates the top four TCS candidate combinations across 5 chronological expanding folds to test temporal consistency, worst-fold risk, and return spreads.",
        "",
        "## Candidate Tournament Ranking & Final Verdicts",
        "",
    ]

    if not df_ranking.empty:
        cols = ["candidate_id", "target_name", "model", "stability_score", "pos_auc_folds", "pos_mcc_folds", "pos_spread_folds", "mean_roc_auc", "std_roc_auc", "min_roc_auc", "mean_mcc", "mean_return_spread_pct", "verdict"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_ranking.iterrows():
            vals = [f"{r[c]:.4f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("## Fold-by-Fold Performance Matrix")
    lines.append("")
    if not df_fold.empty:
        cols = ["candidate_id", "fold", "roc_auc", "pr_auc", "mcc", "f1", "ppr", "mean_realized_ret_buy", "mean_realized_ret_sell", "return_spread", "cum_strategy_return", "max_drawdown"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_fold.iterrows():
            vals = [f"{r[c]:.4f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")

    filepath.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.perf_counter()
    logger.info("Executing Mission 15 Step 3 TCS Candidate Tournament...")
    res = run_tcs_candidate_tournament()
    elapsed = time.perf_counter() - t0
    logger.info("TCS Candidate Tournament completed in %.2f seconds.", elapsed)
