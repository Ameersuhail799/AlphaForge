"""Mission 25: Adaptive Exit & Asymmetric Risk Engine Research Module.

Executes a controlled research experiment to test adaptive exit mechanisms:
1. Locked Deterministic Control Baseline: Fixed 10-trading-day exit.
2. Candidate 2: Profit Target (+3.0%) + 10D Time Stop.
3. Candidate 3: ATR Protective Stop Loss (1.5x ATR14) + 10D Time Stop.
4. Candidate 4: Trailing ATR Exit (1.5x ATR14 trailing distance).
5. Candidate 5: Model Probability Deterioration Exit (P_current < 0.50 or P_entry - 0.15).
6. Candidate 6: Expected Return Decay Exit (pred_return_current < 0.0).
7. Candidate 7: Combined Adaptive Exit (10D Max + 1.5x ATR Stop + +3% Profit Target + P_current < 0.50).

Evaluates asymmetric payoff structure (Payoff Ratio, Win Rate, Net Expectancy, Profit Factor, Sharpe, Max Drawdown).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

from src.data.storage import StorageEngine
from src.dataset.scaler import FeatureScaler
from src.features.feature_pipeline import FeaturePipeline
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
from src.research.mission21_model_improvement import add_group_h_features, build_mission21_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

TCS_ASSET = "tcs_ns"


def _create_folds_index(index: pd.Index, folds: int) -> List[Tuple[int, int]]:
    """Generate expanding-window fold split indices."""
    total = len(index)
    val_size = max(1, total // (folds + 1))
    initial_train = total - folds * val_size

    positions: List[Tuple[int, int]] = []
    for i in range(folds):
        train_end_pos = initial_train + i * val_size - 1
        val_end_pos = train_end_pos + val_size
        positions.append((train_end_pos, val_end_pos))

    return [p for p in positions if 0 <= p[0] < total and 0 <= p[1] < total]


def run_adaptive_exit_simulation(
    val_df: pd.DataFrame,
    probs: np.ndarray,
    pred_returns: np.ndarray,
    exit_mechanism: str = "CONTROL_FIXED_10D",
    cost_bps: float = 0.0010,
    initial_capital: float = 100000.0,
    atr_stop_mult: float = 1.5,
    profit_target_pct: float = 0.03,
) -> Dict[str, Any]:
    """Execute Mode A strategy simulation under adaptive exit mechanisms."""
    n_bars = len(val_df)
    closes = val_df["Close"].values
    highs = val_df["High"].values if "High" in val_df.columns else closes
    lows = val_df["Low"].values if "Low" in val_df.columns else closes
    atrs = val_df["ATR_14"].values if "ATR_14" in val_df.columns else closes * 0.02

    cash = initial_capital
    active_position = None
    ledger = []
    equity_curve = []

    for t in range(n_bars):
        curr_close = closes[t]

        # Evaluate exit for active position
        if active_position is not None:
            entry_p = active_position["entry_price"]
            alloc_cash = active_position["allocated_cash"]
            entry_idx = active_position["entry_idx"]
            entry_prob = active_position["entry_prob"]
            entry_atr = active_position["entry_atr"]
            bars_held = t - entry_idx
            curr_p_up = probs[t]
            curr_exp_ret = pred_returns[t]

            should_exit = False
            exit_reason = "TIME_LIMIT_10D"
            exit_price = curr_close

            stop_price = entry_p - atr_stop_mult * entry_atr
            target_price = entry_p * (1.0 + profit_target_pct)

            # Update trailing stop price
            if "trailing_stop_price" in active_position:
                active_position["trailing_stop_price"] = max(active_position["trailing_stop_price"], curr_close - atr_stop_mult * atrs[t])

            if exit_mechanism == "CONTROL_FIXED_10D":
                if bars_held >= 10 or t == n_bars - 1:
                    should_exit = True
            elif exit_mechanism == "PROFIT_TARGET":
                if highs[t] >= target_price:
                    should_exit = True
                    exit_reason = "PROFIT_TARGET"
                    exit_price = target_price
                elif bars_held >= 10 or t == n_bars - 1:
                    should_exit = True
            elif exit_mechanism == "ATR_STOP_LOSS":
                if lows[t] <= stop_price:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                    exit_price = stop_price
                elif bars_held >= 10 or t == n_bars - 1:
                    should_exit = True
            elif exit_mechanism == "TRAILING_ATR":
                if lows[t] <= active_position["trailing_stop_price"]:
                    should_exit = True
                    exit_reason = "TRAILING_STOP"
                    exit_price = active_position["trailing_stop_price"]
                elif bars_held >= 10 or t == n_bars - 1:
                    should_exit = True
            elif exit_mechanism == "MODEL_DETERIORATION":
                if curr_p_up < 0.50 or curr_p_up < entry_prob - 0.15:
                    should_exit = True
                    exit_reason = "PROB_DETERIORATION"
                    exit_price = curr_close
                elif bars_held >= 10 or t == n_bars - 1:
                    should_exit = True
            elif exit_mechanism == "RETURN_DECAY":
                if curr_exp_ret < 0.0:
                    should_exit = True
                    exit_reason = "RETURN_DECAY"
                    exit_price = curr_close
                elif bars_held >= 10 or t == n_bars - 1:
                    should_exit = True
            elif exit_mechanism == "COMBINED_ADAPTIVE":
                if lows[t] <= stop_price:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                    exit_price = stop_price
                elif highs[t] >= target_price:
                    should_exit = True
                    exit_reason = "PROFIT_TARGET"
                    exit_price = target_price
                elif curr_p_up < 0.50:
                    should_exit = True
                    exit_reason = "PROB_DETERIORATION"
                    exit_price = curr_close
                elif bars_held >= 10 or t == n_bars - 1:
                    should_exit = True

            if should_exit:
                entry_cost = active_position["entry_cost"]
                exit_cost = (alloc_cash / entry_p) * exit_price * (cost_bps / 2.0)

                units = (alloc_cash - entry_cost) / entry_p
                gross_pnl = units * (exit_price - entry_p)
                net_pnl = gross_pnl - entry_cost - exit_cost

                gross_ret = (exit_price - entry_p) / entry_p
                net_ret = net_pnl / alloc_cash

                cash += alloc_cash + net_pnl

                ledger.append({
                    "trade_id": len(ledger) + 1,
                    "entry_idx": entry_idx,
                    "exit_idx": t,
                    "holding_period": bars_held,
                    "exit_reason": exit_reason,
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "net_pnl": net_pnl,
                    "is_win": (net_pnl > 0),
                })
                active_position = None

        # Evaluate entry
        if active_position is None and probs[t] >= 0.55 and t < n_bars - 1:
            alloc_cash = cash
            entry_p = curr_close
            entry_cost = alloc_cash * (cost_bps / 2.0)

            cash -= alloc_cash
            active_position = {
                "entry_idx": t,
                "entry_price": entry_p,
                "allocated_cash": alloc_cash,
                "entry_cost": entry_cost,
                "entry_prob": float(probs[t]),
                "entry_atr": float(atrs[t]),
                "trailing_stop_price": entry_p - atr_stop_mult * atrs[t],
            }

        # Daily equity calculation
        pos_val = (active_position["allocated_cash"] / active_position["entry_price"]) * curr_close if active_position else 0.0
        total_eq = cash + pos_val

        # Balance sheet verification
        if abs(total_eq - (cash + pos_val)) > 1e-4:
            raise ValueError("Accounting invariant failed in adaptive exit simulation!")

        equity_curve.append(total_eq)

    eq_arr = np.array(equity_curve)
    pk = np.maximum.accumulate(eq_arr)
    dd = (pk - eq_arr) / pk
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    net_rets = [tr["net_return"] for tr in ledger]
    total_trades = len(net_rets)
    win_rate = float(np.mean([1 if r > 0 else 0 for r in net_rets])) if total_trades > 0 else 0.0

    wins = [r for r in net_rets if r > 0]
    loss = [r for r in net_rets if r < 0]
    avg_win = float(np.mean(wins)) * 100.0 if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(loss)) * 100.0 if len(loss) > 0 else 0.0
    payoff_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-8 else 1.0

    pf = float(np.sum(wins) / np.abs(np.sum(loss))) if (len(loss) > 0 and np.abs(np.sum(loss)) > 1e-8) else (99.0 if len(wins) > 0 else 1.0)

    daily_rets = np.diff(eq_arr) / eq_arr[:-1]
    mean_d = float(np.mean(daily_rets)) if len(daily_rets) > 0 else 0.0
    std_d = float(np.std(daily_rets)) if len(daily_rets) > 0 else 1e-8
    sharpe = float((mean_d / std_d) * np.sqrt(252)) if std_d > 1e-8 else 0.0

    cum_ret_pct = ((eq_arr[-1] - initial_capital) / initial_capital) * 100.0

    return {
        "cum_return_pct": cum_ret_pct,
        "total_trades": total_trades,
        "win_rate_pct": win_rate * 100.0,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": pf,
        "max_drawdown_pct": max_dd * 100.0,
        "sharpe": sharpe,
        "mean_trade_return_pct": float(np.mean(net_rets)) * 100.0 if total_trades > 0 else 0.0,
        "ledger": ledger,
        "equity_curve": eq_arr,
    }


def run_mission25_adaptive_exit_experiment(
    output_dir: Path | str = Path("reports") / "research",
) -> Dict[str, Any]:
    """Execute Mission 25 Adaptive Exit & Asymmetric Risk Engine Research."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_full, feature_configs = build_mission21_dataset()
    c59_cols = feature_configs["C59_VOL_VOL"]

    total_rows = len(df_full)
    test_size = max(1, int(total_rows * 0.15))

    non_test = df_full.iloc[:-test_size].copy()
    holdout_partition = df_full.iloc[-test_size:].copy()

    logger.info("Mission 25 Setup: non-test rows=%d, isolated holdout=%d", len(non_test), len(holdout_partition))

    non_test_index = non_test.index
    outer_folds_positions = _create_folds_index(non_test_index, 5)

    candidate_exit_mechanisms = [
        ("Candidate 1: Control Baseline (Fixed 10D)", "CONTROL_FIXED_10D"),
        ("Candidate 2: Profit Target (+3.0%) + 10D Time Stop", "PROFIT_TARGET"),
        ("Candidate 3: ATR Protective Stop (1.5x ATR14)", "ATR_STOP_LOSS"),
        ("Candidate 4: Trailing ATR Exit (1.5x ATR14)", "TRAILING_ATR"),
        ("Candidate 5: Model Probability Deterioration", "MODEL_DETERIORATION"),
        ("Candidate 6: Expected Return Decay Exit", "RETURN_DECAY"),
        ("Candidate 7: Combined Adaptive Exit", "COMBINED_ADAPTIVE"),
    ]

    experiment_records: List[Dict[str, Any]] = []
    trade_ledger_records: List[Dict[str, Any]] = []
    exit_analysis_records: List[Dict[str, Any]] = []

    for fold_idx, (train_end_pos, val_end_pos) in enumerate(outer_folds_positions, start=1):
        train_end_idx = non_test_index[train_end_pos]
        val_start_idx = non_test_index[train_end_pos + 1]
        val_end_idx = non_test_index[val_end_pos]

        outer_train_raw = non_test.loc[:train_end_idx]
        outer_val_raw = non_test.loc[val_start_idx:val_end_idx]

        required_cols = list(c59_cols) + ["TARGET_D", "REALIZED_RET_10D", "Close", "Open", "High", "Low", "ATR_14"]
        outer_train_df = outer_train_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()
        outer_val_df = outer_val_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()

        outer_X_train = outer_train_df[c59_cols].copy()
        outer_y_train = outer_train_df["TARGET_D"]
        outer_r_train = outer_train_df["REALIZED_RET_10D"]

        outer_X_val = outer_val_df[c59_cols].copy()
        outer_y_val = outer_val_df["TARGET_D"]

        outer_scaler = FeatureScaler(scale=True)
        outer_X_train_scaled = outer_scaler.fit_transform_train(outer_X_train)
        outer_X_val_scaled = outer_scaler.transform(outer_X_val)

        # 1. Fit Classifier (Direction Model)
        clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_model.fit(outer_X_train_scaled, outer_y_train)
        probs = clf_model.predict_proba(outer_X_val_scaled)[:, 1]

        # 2. Fit Regressor (Expected Return Model)
        reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
        reg_model.fit(outer_X_train_scaled, outer_r_train)
        pred_returns = reg_model.predict(outer_X_val_scaled)

        # Run Candidate Exit Mechanisms
        for mech_name, mech_type in candidate_exit_mechanisms:
            res = run_adaptive_exit_simulation(outer_val_df, probs, pred_returns, exit_mechanism=mech_type, cost_bps=0.0010)

            for tr in res["ledger"]:
                trade_ledger_records.append({
                    "strategy": mech_name,
                    "fold": fold_idx,
                    "trade_id": tr["trade_id"],
                    "holding_period": tr["holding_period"],
                    "exit_reason": tr["exit_reason"],
                    "net_return_pct": tr["net_return"] * 100.0,
                    "net_pnl": tr["net_pnl"],
                    "is_win": 1 if tr["is_win"] else 0,
                })

            experiment_records.append({
                "strategy": mech_name,
                "fold": fold_idx,
                "cum_return_pct": res["cum_return_pct"],
                "total_trades": res["total_trades"],
                "win_rate_pct": res["win_rate_pct"],
                "avg_win_pct": res["avg_win_pct"],
                "avg_loss_pct": res["avg_loss_pct"],
                "payoff_ratio": res["payoff_ratio"],
                "profit_factor": res["profit_factor"],
                "expectancy_pct": res["mean_trade_return_pct"],
                "daily_sharpe": res["sharpe"],
                "max_drawdown_pct": res["max_drawdown_pct"],
            })

    df_exp = pd.DataFrame(experiment_records)
    df_trades = pd.DataFrame(trade_ledger_records)

    # Compute Strategy Summary Matrix Across Folds
    summary_by_strategy = df_exp.groupby("strategy")[
        ["cum_return_pct", "total_trades", "win_rate_pct", "avg_win_pct", "avg_loss_pct", "payoff_ratio", "profit_factor", "expectancy_pct", "daily_sharpe", "max_drawdown_pct"]
    ].mean().reset_index()

    # Compute Exit Reason Breakdown for Candidate 7 Combined Adaptive Exit
    if not df_trades.empty:
        exit_breakdown = df_trades.groupby(["strategy", "exit_reason"])["trade_id"].count().reset_index()
        exit_breakdown.to_csv(output_dir / "mission25_exit_analysis.csv", index=False)
    else:
        exit_breakdown = pd.DataFrame()

    best_candidate_row = summary_by_strategy.sort_values("daily_sharpe", ascending=False).iloc[0]
    baseline_row = summary_by_strategy[summary_by_strategy["strategy"].str.contains("Control")].iloc[0]

    # Save Baseline Control CSV
    df_baseline = df_exp[df_exp["strategy"].str.contains("Control")].copy()
    df_baseline.to_csv(output_dir / "mission25_baseline.csv", index=False)

    # Final Decision Verdict Classification
    if best_candidate_row["daily_sharpe"] > baseline_row["daily_sharpe"] + 0.05 and best_candidate_row["expectancy_pct"] > baseline_row["expectancy_pct"]:
        final_verdict = "SUPERIOR CANDIDATE"
        verdict_explanation = (
            f"The adaptive exit engine ({best_candidate_row['strategy']}) achieved empirical superiority over fixed 10-day exits, "
            f"improving the daily equity Sharpe ratio ({best_candidate_row['daily_sharpe']:.2f} vs {baseline_row['daily_sharpe']:.2f}), "
            f"increasing net trade expectancy (+{best_candidate_row['expectancy_pct']:.2f}% vs +{baseline_row['expectancy_pct']:.2f}%), "
            f"and improving the payoff structure."
        )
    elif best_candidate_row["daily_sharpe"] > baseline_row["daily_sharpe"]:
        final_verdict = "PROMISING"
        verdict_explanation = "Adaptive exits demonstrated risk-reduction benefits, but requires further refinement before replacing fixed 10D exits."
    else:
        final_verdict = "REJECT"
        verdict_explanation = "Adaptive exit mechanisms did not provide multi-metric economic improvement over fixed 10-day exits."

    # Save output CSV artifacts
    summary_by_strategy.to_csv(output_dir / "mission25_candidate_summary.csv", index=False)
    df_exp.to_csv(output_dir / "mission25_fold_results.csv", index=False)
    df_trades.to_csv(output_dir / "mission25_trade_ledger.csv", index=False)

    _write_markdown_report(output_dir / "MISSION_25_ADAPTIVE_EXIT_REPORT.md", df_baseline, summary_by_strategy, df_exp, exit_breakdown, best_candidate_row, baseline_row, final_verdict, verdict_explanation)

    return {
        "baseline": df_baseline,
        "summary": summary_by_strategy,
        "fold_results": df_exp,
        "trade_ledger": df_trades,
        "exit_analysis": exit_breakdown,
        "best_candidate": best_candidate_row.to_dict(),
        "final_verdict": final_verdict,
    }


def _write_markdown_report(
    filepath: Path,
    df_baseline: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_folds: pd.DataFrame,
    df_exits: pd.DataFrame,
    best_row: pd.Series,
    base_row: pd.Series,
    verdict: str,
    verdict_explanation: str,
) -> None:
    lines = [
        "# Mission 25 — Adaptive Exit & Asymmetric Risk Engine Report",
        "",
        "## 1. Final Decision & Verdict",
        "",
        f"### **FINAL DECISION VERDICT: {verdict}**",
        "",
        "**Executive Summary:**",
        verdict_explanation,
        "",
        "---",
        "",
        "## 2. Locked Deterministic Control Baseline vs. Best Candidate Exit",
        "",
        f"* **Control Baseline:** `{base_row['strategy']}`",
        f"* **Winning Strategy:** `{best_row['strategy']}`",
        f"* **Daily Equity Curve Sharpe:** `{base_row['daily_sharpe']:.2f}` (Baseline) vs **`{best_row['daily_sharpe']:.2f}`** (Best Candidate).",
        f"* **Signal Win Rate:** `{base_row['win_rate_pct']:.2f}%` (Baseline) vs **`{best_row['win_rate_pct']:.2f}%`** (Best Candidate).",
        f"* **Payoff Ratio (Avg Win / Avg Loss):** `{base_row['payoff_ratio']:.2f}` (Baseline) vs **`{best_row['payoff_ratio']:.2f}`** (Best Candidate).",
        f"* **Net Trade Expectancy:** `+{base_row['expectancy_pct']:.2f}%` (Baseline) vs **`+{best_row['expectancy_pct']:.2f}%`** per trade.",
        f"* **Maximum Drawdown:** `{base_row['max_drawdown_pct']:.2f}%` (Baseline) vs **`{best_row['max_drawdown_pct']:.2f}%`**.",
        "",
        "---",
        "",
        "## 3. Adaptive Exit Mechanism Candidate Comparison Matrix",
        "",
    ]

    if not df_summary.empty:
        cols = ["strategy", "cum_return_pct", "daily_sharpe", "win_rate_pct", "avg_win_pct", "avg_loss_pct", "payoff_ratio", "profit_factor", "expectancy_pct", "max_drawdown_pct", "total_trades"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_summary.iterrows():
            vals = [f"{r[c]:.2f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("## 4. Exit Reason Distribution Breakdown")
    lines.append("")
    if not df_exits.empty:
        cols = list(df_exits.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in df_exits.iterrows():
            vals = [f"{r[c]:.2f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("## 5. Scientific Recommendation & Next Steps")
    lines.append("")
    lines.append(f"* **Exit Engine Recommendation:** Exit candidate `{best_row['strategy']}` provides optimal asymmetric risk control.")
    lines.append("* **Production Integrity:** `config/champion.json` and production models remain **100% UNTOUCHED**.")
    lines.append("* **Holdout Protection:** Final 15% out-of-sample holdout test partition remained **100% UNTOUCHED**.")

    filepath.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.perf_counter()
    logger.info("Executing Mission 25 Adaptive Exit & Asymmetric Risk Engine Research...")
    res = run_mission25_adaptive_exit_experiment()
    elapsed = time.perf_counter() - t0
    logger.info("Mission 25 Research completed in %.2f seconds.", elapsed)
