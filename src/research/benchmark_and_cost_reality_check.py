"""Measurement Task: Benchmark and Realistic Cost Reality Check for AlphaForge.

Evaluates:
1. Production Champion Strategy under old 10bps (0.10%) flat round-trip cost assumption.
2. Production Champion Strategy under realistic 2026 NSE delivery-equity transaction costs:
   - STT: 0.1% on buy leg, 0.1% on sell leg (0.20% round-trip total)
   - Stamp Duty: 0.015% on buy leg only
   - Exchange Transaction Charge + SEBI Fee: 0.003% per leg + 18% GST (0.00354% per leg)
   - DP (Depository Participant) Charge: Flat INR 15.93 per stock per sell order (incl. GST)
3. Equal-Weight Buy-and-Hold Benchmark for the exact same 5-stock universe over identical evaluation dates,
   applying realistic entry costs once (STT 0.1% + Stamp Duty 0.015% + Exchange Fee + SEBI + GST = 0.11864% entry fee).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.dataset.scaler import FeatureScaler
from src.research.mission21_model_improvement import C57_FEATURES
from src.research.mission27_cross_asset_generalization import ASSET_UNIVERSE, _create_folds_index, build_asset_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

C59_COLS = C57_FEATURES + ["RANGE_COMPRESSION_EXP", "VOLUME_BREAKOUT_CONFIRM", "TREND_VOL_INTERACTION"]


def calculate_nse_delivery_cost(
    trade_value: float,
    is_buy: bool,
    stt_rate: float = 0.0010,
    stamp_duty_rate: float = 0.00015,
    exchange_charge_rate: float = 0.00003,
    sebi_fee_rate: float = 0.000001,
    gst_rate: float = 0.18,
    dp_charge_flat: float = 15.93,
) -> float:
    """Calculate exact 2026 NSE delivery equity transaction cost for a single leg.

    Rates:
    - STT: 0.10% on Buy, 0.10% on Sell
    - Stamp Duty: 0.015% on Buy only
    - Exchange Charge: 0.003% per leg + 18% GST (0.00354% per leg)
    - SEBI Turnover Fee: 0.0001% per leg
    - DP Charge: Flat INR 15.93 per stock per sell order (incl. GST)
    """
    if is_buy:
        stt = trade_value * stt_rate
        stamp_duty = trade_value * stamp_duty_rate
        exch_charge = trade_value * exchange_charge_rate
        gst = exch_charge * gst_rate
        sebi = trade_value * sebi_fee_rate
        dp = 0.0
    else:
        stt = trade_value * stt_rate
        stamp_duty = 0.0
        exch_charge = trade_value * exchange_charge_rate
        gst = exch_charge * gst_rate
        sebi = trade_value * sebi_fee_rate
        dp = dp_charge_flat

    return stt + stamp_duty + exch_charge + gst + sebi + dp


def run_benchmark_measurement(output_dir: Path | str = Path("reports") / "validation") -> Dict[str, Any]:
    """Run exact measurement benchmark comparing Old 10bps vs Realistic Costs vs Buy-and-Hold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets across 5 liquid equities: %s", ASSET_UNIVERSE)
    asset_datasets = {a: build_asset_dataset(a) for a in ASSET_UNIVERSE}
    daily_signals_dict: Dict[str, pd.Series] = {}

    # Generate walk-forward signals using exact champion configuration
    logger.info("Generating 5-fold walk-forward signals using Champion ML models...")
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

            # Champion Dual-Agreement Signal Rule
            sigs = ((probs >= 0.55) & (pred_returns > 0.01)).astype(int)
            asset_signals.loc[outer_val_df.index] = sigs
        daily_signals_dict[asset] = asset_signals

    # Align common dates across asset universe
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
    logger.info("Evaluation period: %s to %s (%d trading days)", start_date_str, end_date_str, len(common_dates_eval))

    def _simulate_champion(use_realistic_costs: bool) -> Dict[str, Any]:
        cash = 100000.0
        initial_cap = 100000.0
        n_assets = len(ASSET_UNIVERSE)
        alloc_per_asset = 1.0 / n_assets
        asset_positions = {a: None for a in ASSET_UNIVERSE}
        portfolio_equity = []
        total_trades = 0
        trade_returns = []

        for dt in common_dates_eval:
            for a in ASSET_UNIVERSE:
                df_a = asset_datasets[a]
                if dt in df_a.index:
                    close_p = df_a.loc[dt, "Close"]
                    sig = daily_signals_dict[a].loc[dt] if dt in daily_signals_dict[a].index else 0

                    pos = asset_positions[a]
                    if pos is not None:
                        bars_held = (dt - pos["entry_date"]).days
                        if bars_held >= 14 or dt == common_dates_eval[-1]:
                            sell_val = pos["units"] * close_p
                            if use_realistic_costs:
                                exit_cost = calculate_nse_delivery_cost(sell_val, is_buy=False)
                            else:
                                exit_cost = sell_val * 0.0005

                            net_pnl = (sell_val - exit_cost) - pos["total_entry_outflow"]
                            cash += pos["alloc"] + net_pnl
                            total_trades += 1
                            trade_returns.append(net_pnl / pos["alloc"])
                            asset_positions[a] = None

                    if asset_positions[a] is None and sig == 1:
                        alloc_amt = cash * alloc_per_asset
                        if use_realistic_costs:
                            entry_cost = calculate_nse_delivery_cost(alloc_amt, is_buy=True)
                        else:
                            entry_cost = alloc_amt * 0.0005

                        units = (alloc_amt - entry_cost) / close_p
                        cash -= alloc_amt
                        asset_positions[a] = {
                            "entry_date": dt,
                            "entry_price": close_p,
                            "alloc": alloc_amt,
                            "units": units,
                            "entry_cost": entry_cost,
                            "total_entry_outflow": alloc_amt,
                        }

            pos_val_sum = 0.0
            for a in ASSET_UNIVERSE:
                if asset_positions[a] is not None:
                    df_a = asset_datasets[a]
                    curr_p = df_a.loc[dt, "Close"] if dt in df_a.index else asset_positions[a]["entry_price"]
                    pos_val_sum += asset_positions[a]["units"] * curr_p
            eq = cash + pos_val_sum
            portfolio_equity.append(eq)

        eq_arr = np.array(portfolio_equity)
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

        win_rate = float(np.mean([1 if r > 0 else 0 for r in trade_returns])) * 100.0 if trade_returns else 0.0
        expectancy_pct = float(np.mean(trade_returns)) * 100.0 if trade_returns else 0.0

        return {
            "total_return_pct": round(float(total_ret_pct), 2),
            "cagr_pct": round(float(cagr), 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "expectancy_pct": round(expectancy_pct, 2),
        }

    def _simulate_buy_and_hold() -> Dict[str, Any]:
        initial_cap = 100000.0
        n_assets = len(ASSET_UNIVERSE)
        alloc_per_asset = initial_cap / n_assets
        start_dt = common_dates_eval[0]
        units_dict = {}

        for a in ASSET_UNIVERSE:
            df_a = asset_datasets[a]
            start_p = df_a.loc[start_dt, "Close"]
            entry_cost = calculate_nse_delivery_cost(alloc_per_asset, is_buy=True)
            units = (alloc_per_asset - entry_cost) / start_p
            units_dict[a] = units

        bh_equity = []
        for dt in common_dates_eval:
            eq = 0.0
            for a in ASSET_UNIVERSE:
                df_a = asset_datasets[a]
                curr_p = df_a.loc[dt, "Close"] if dt in df_a.index else df_a.loc[start_dt, "Close"]
                eq += units_dict[a] * curr_p
            bh_equity.append(eq)

        eq_arr = np.array(bh_equity)
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
        }

    res_10bps = _simulate_champion(use_realistic_costs=False)
    res_realistic = _simulate_champion(use_realistic_costs=True)
    res_buy_hold = _simulate_buy_and_hold()

    summary_results = {
        "evaluation_period": f"{start_date_str} to {end_date_str}",
        "trading_days": len(common_dates_eval),
        "champion_10bps": res_10bps,
        "champion_realistic": res_realistic,
        "buy_and_hold": res_buy_hold,
    }

    # Save summary report CSV
    df_compare = pd.DataFrame([
        {"strategy": "Champion Strategy (Old 10bps)", **res_10bps},
        {"strategy": "Champion Strategy (Realistic Costs)", **res_realistic},
        {"strategy": "Equal-Weight Buy-and-Hold", **res_buy_hold},
    ])
    df_compare.to_csv(output_dir / "benchmark_comparison.csv", index=False)
    logger.info("Measurement benchmark complete. Comparison saved to %s", output_dir / "benchmark_comparison.csv")

    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measurement Benchmark & Cost Reality Check")
    parser.add_argument("--output-dir", default="reports/validation", help="Output directory")
    args = parser.parse_args()

    res = run_benchmark_measurement(args.output_dir)
    print("\n" + "=" * 60)
    print("MEASUREMENT BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Period: {res['evaluation_period']} ({res['trading_days']} trading days)")
    print(f"Champion (Old 10bps): Total Return = +{res['champion_10bps']['total_return_pct']}%, CAGR = {res['champion_10bps']['cagr_pct']}%, Sharpe = {res['champion_10bps']['sharpe']}, Max DD = {res['champion_10bps']['max_dd_pct']}%")
    print(f"Champion (Realistic Costs): Total Return = +{res['champion_realistic']['total_return_pct']}%, CAGR = {res['champion_realistic']['cagr_pct']}%, Sharpe = {res['champion_realistic']['sharpe']}, Max DD = {res['champion_realistic']['max_dd_pct']}%")
    print(f"Equal-Weight Buy & Hold: Total Return = +{res['buy_and_hold']['total_return_pct']}%, CAGR = {res['buy_and_hold']['cagr_pct']}%, Sharpe = {res['buy_and_hold']['sharpe']}, Max DD = {res['buy_and_hold']['max_dd_pct']}%")
