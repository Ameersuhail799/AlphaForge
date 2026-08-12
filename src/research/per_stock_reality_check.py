"""Per-Stock Benchmark & Cost Reality Check Research Module.

Executes single-asset evaluation of the Production Champion Strategy vs. Buy-and-Hold
independently for each stock (TCS, INFY, RELIANCE, ICICIBANK, HDFCBANK) under 100% capital allocation
and realistic 2026 NSE delivery transaction costs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.dataset.scaler import FeatureScaler
from src.research.benchmark_and_cost_reality_check import (
    ASSET_UNIVERSE,
    C59_COLS,
    calculate_nse_delivery_cost,
)
from src.research.mission27_cross_asset_generalization import _create_folds_index, build_asset_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_per_stock_measurement(output_dir: Path | str = Path("reports") / "validation") -> Dict[str, Any]:
    """Execute per-stock single-asset measurement audit."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building datasets across asset universe: %s", ASSET_UNIVERSE)
    asset_datasets = {a: build_asset_dataset(a) for a in ASSET_UNIVERSE}
    daily_signals_dict: Dict[str, pd.Series] = {}

    for asset in ASSET_UNIVERSE:
        df_full = asset_datasets[asset]
        test_size = max(1, int(len(df_full) * 0.15))
        non_test = df_full.iloc[:-test_size].copy()
        non_test_index = non_test.index
        outer_folds_positions = _create_folds_index(non_test_index, 5)

        asset_signals = pd.Series(0, index=df_full.index)
        for fold_idx, (train_end_pos, val_end_pos) in enumerate(outer_folds_positions, start=1):
            train_end_idx = non_test_index[train_end_pos]
            val_start_idx = non_test_index[train_end_pos + 1]
            val_end_idx = non_test_index[val_end_pos]

            outer_train_raw = non_test.loc[:train_end_idx]
            outer_val_raw = non_test.loc[val_start_idx:val_end_idx]

            required_cols = list(C59_COLS) + ["TARGET_D", "REALIZED_RET_10D", "Close", "Open", "High", "Low"]
            outer_train_df = outer_train_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()
            outer_val_df = outer_val_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()

            outer_X_train = outer_train_df[C59_COLS].copy()
            outer_y_train = outer_train_df["TARGET_D"]
            outer_r_train = outer_train_df["REALIZED_RET_10D"]

            outer_X_val = outer_val_df[C59_COLS].copy()

            outer_scaler = FeatureScaler(scale=True)
            outer_X_train_scaled = outer_scaler.fit_transform_train(outer_X_train)
            outer_X_val_scaled = outer_scaler.transform(outer_X_val)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(outer_X_train_scaled, outer_y_train)
            probs = clf.predict_proba(outer_X_val_scaled)[:, 1]

            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(outer_X_train_scaled, outer_r_train)
            pred_returns = reg.predict(outer_X_val_scaled)

            sigs = ((probs >= 0.55) & (pred_returns > 0.01)).astype(int)
            asset_signals.loc[outer_val_df.index] = sigs
        daily_signals_dict[asset] = asset_signals

    common_dates = None
    for a in ASSET_UNIVERSE:
        if common_dates is None:
            common_dates = asset_datasets[a].index
        else:
            common_dates = common_dates.intersection(asset_datasets[a].index)
    common_dates = common_dates.sort_values()

    eval_dates = [dt for dt in common_dates if any(daily_signals_dict[a].loc[dt] != 0 for a in ASSET_UNIVERSE if dt in daily_signals_dict[a].index)]
    min_eval_dt = min(eval_dates) if eval_dates else common_dates[0]
    common_dates_eval = common_dates[common_dates >= min_eval_dt]

    start_date_str = str(common_dates_eval[0])[:10]
    end_date_str = str(common_dates_eval[-1])[:10]

    def _run_single_champion(asset: str, initial_cap: float = 100000.0) -> Dict[str, Any]:
        df_a = asset_datasets[asset]
        sigs_a = daily_signals_dict[asset]
        cash = initial_cap
        active_pos = None
        equity_curve = []
        trade_ledger = []

        for dt in common_dates_eval:
            if dt in df_a.index:
                close_p = df_a.loc[dt, "Close"]
                sig = sigs_a.loc[dt] if dt in sigs_a.index else 0

                if active_pos is not None:
                    bars_held = (dt - active_pos["entry_date"]).days
                    if bars_held >= 14 or dt == common_dates_eval[-1]:
                        sell_val = active_pos["units"] * close_p
                        exit_cost = calculate_nse_delivery_cost(sell_val, is_buy=False)
                        net_pnl = (sell_val - exit_cost) - active_pos["total_entry_outflow"]
                        cash += active_pos["alloc"] + net_pnl
                        trade_ledger.append({
                            "net_pnl": net_pnl,
                            "net_return": net_pnl / active_pos["alloc"],
                            "is_win": (net_pnl > 0),
                        })
                        active_pos = None

                if active_pos is None and sig == 1:
                    alloc_amt = cash
                    entry_cost = calculate_nse_delivery_cost(alloc_amt, is_buy=True)
                    units = (alloc_amt - entry_cost) / close_p
                    cash -= alloc_amt
                    active_pos = {
                        "entry_date": dt,
                        "entry_price": close_p,
                        "alloc": alloc_amt,
                        "units": units,
                        "total_entry_outflow": alloc_amt,
                    }

            pos_val = active_pos["units"] * df_a.loc[dt, "Close"] if (active_pos is not None and dt in df_a.index) else 0.0
            eq = cash + pos_val
            equity_curve.append(eq)

        eq_arr = np.array(equity_curve)
        total_ret_pct = ((eq_arr[-1] - initial_cap) / initial_cap) * 100.0
        n_years = len(common_dates_eval) / 252.0
        cagr = ((eq_arr[-1] / initial_cap) ** (1.0 / n_years) - 1.0) * 100.0
        pk = np.maximum.accumulate(eq_arr)
        dd = (pk - eq_arr) / pk
        max_dd_pct = float(np.max(dd)) * 100.0

        daily_rets = np.diff(eq_arr) / eq_arr[:-1]
        mean_d = float(np.mean(daily_rets)) if len(daily_rets) > 0 else 0.0
        std_d = float(np.std(daily_rets)) if len(daily_rets) > 0 else 1e-8
        sharpe = float((mean_d / std_d) * np.sqrt(252)) if std_d > 1e-8 else 0.0

        downside_std = float(np.std(daily_rets[daily_rets < 0])) if np.sum(daily_rets < 0) > 0 else 1e-8
        sortino = float((mean_d / downside_std) * np.sqrt(252)) if downside_std > 1e-8 else 0.0

        total_trades = len(trade_ledger)
        win_rate = float(np.mean([1 if tr["is_win"] else 0 for tr in trade_ledger])) * 100.0 if total_trades > 0 else 0.0

        return {
            "total_return_pct": round(float(total_ret_pct), 2),
            "cagr_pct": round(float(cagr), 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
        }

    def _run_single_buy_hold(asset: str, initial_cap: float = 100000.0) -> Dict[str, Any]:
        df_a = asset_datasets[asset]
        start_dt = common_dates_eval[0]
        start_p = df_a.loc[start_dt, "Close"]

        entry_cost = calculate_nse_delivery_cost(initial_cap, is_buy=True)
        units = (initial_cap - entry_cost) / start_p

        equity_curve = []
        for dt in common_dates_eval:
            curr_p = df_a.loc[dt, "Close"] if dt in df_a.index else start_p
            eq = units * curr_p
            equity_curve.append(eq)

        eq_arr = np.array(equity_curve)
        total_ret_pct = ((eq_arr[-1] - initial_cap) / initial_cap) * 100.0
        n_years = len(common_dates_eval) / 252.0
        cagr = ((eq_arr[-1] / initial_cap) ** (1.0 / n_years) - 1.0) * 100.0
        pk = np.maximum.accumulate(eq_arr)
        dd = (pk - eq_arr) / pk
        max_dd_pct = float(np.max(dd)) * 100.0

        daily_rets = np.diff(eq_arr) / eq_arr[:-1]
        mean_d = float(np.mean(daily_rets)) if len(daily_rets) > 0 else 0.0
        std_d = float(np.std(daily_rets)) if len(daily_rets) > 0 else 1e-8
        sharpe = float((mean_d / std_d) * np.sqrt(252)) if std_d > 1e-8 else 0.0

        downside_std = float(np.std(daily_rets[daily_rets < 0])) if np.sum(daily_rets < 0) > 0 else 1e-8
        sortino = float((mean_d / downside_std) * np.sqrt(252)) if downside_std > 1e-8 else 0.0

        return {
            "total_return_pct": round(float(total_ret_pct), 2),
            "cagr_pct": round(float(cagr), 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "total_trades": 1,
            "win_rate_pct": 100.0 if total_ret_pct > 0 else 0.0,
        }

    per_stock_data = {}
    for asset in ASSET_UNIVERSE:
        ch = _run_single_champion(asset)
        bh = _run_single_buy_hold(asset)
        cagr_diff = round(ch["cagr_pct"] - bh["cagr_pct"], 2)
        sharpe_diff = round(ch["sharpe"] - bh["sharpe"], 2)

        verdict = "NO"
        if cagr_diff > 1.0 and sharpe_diff > 0.1:
            verdict = "YES"
        elif abs(cagr_diff) <= 1.0 or abs(sharpe_diff) <= 0.05:
            verdict = "MARGINAL"

        per_stock_data[asset] = {
            "champion": ch,
            "buy_hold": bh,
            "cagr_diff": cagr_diff,
            "sharpe_diff": sharpe_diff,
            "verdict": verdict,
        }

    df_rows = []
    for asset, d in per_stock_data.items():
        df_rows.append({
            "asset": asset,
            "champion_cagr_pct": d["champion"]["cagr_pct"],
            "bh_cagr_pct": d["buy_hold"]["cagr_pct"],
            "cagr_diff_pct": d["cagr_diff"],
            "champion_sharpe": d["champion"]["sharpe"],
            "bh_sharpe": d["buy_hold"]["sharpe"],
            "sharpe_diff": d["sharpe_diff"],
            "verdict": d["verdict"],
        })
    df_summary = pd.DataFrame(df_rows).sort_values("cagr_diff_pct", ascending=False)
    df_summary.to_csv(output_dir / "per_stock_summary.csv", index=False)

    return {
        "evaluation_period": f"{start_date_str} to {end_date_str}",
        "trading_days": len(common_dates_eval),
        "per_stock": per_stock_data,
        "summary_table": df_summary.to_dict("records"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Stock Benchmark & Cost Reality Check")
    parser.add_argument("--output-dir", default="reports/validation", help="Output directory")
    args = parser.parse_args()

    res = run_per_stock_measurement(args.output_dir)
    print("\n" + "=" * 60)
    print("PER-STOCK MEASUREMENT AUDIT SUMMARY")
    print("=" * 60)
    for r in res["summary_table"]:
        print(f"Asset: {r['asset'].upper():<12} | Champion CAGR: {r['champion_cagr_pct']:>6.2f}% | BH CAGR: {r['bh_cagr_pct']:>6.2f}% | CAGR Diff: {r['cagr_diff_pct']:>+6.2f}% | Sharpe Diff: {r['sharpe_diff']:>+5.2f} | Verdict: {r['verdict']}")
