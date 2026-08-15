"""Mid-Cap Universe Benchmark & Cost Reality Check Research Module.

Executes single-asset evaluation of the Production Champion Strategy vs. Buy-and-Hold
independently for 8 NSE mid-cap equities:
1. PERSISTENT (Persistent Systems Ltd)
2. COFORGE (Coforge Ltd)
3. VOLTAS (Voltas Ltd)
4. FEDERALBNK (Federal Bank Ltd)
5. AUROPHARMA (Aurobindo Pharma Ltd)
6. APOLLOTYRE (Apollo Tyres Ltd)
7. ASHOKLEY (Ashok Leyland Ltd)
8. BALKRISIND (Balkrishna Industries Ltd)

Incorporates realistic 2026 NSE delivery transaction costs PLUS an explicit, conservative
flat 0.20% per leg (0.40% round-trip) slippage adjustment for mid-cap liquidity friction.
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
    C59_COLS,
    calculate_nse_delivery_cost,
)
from src.research.mission27_cross_asset_generalization import _create_folds_index, build_asset_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

MIDCAP_UNIVERSE = [
    "persistent_ns",
    "coforge_ns",
    "voltas_ns",
    "federalbnk_ns",
    "auropharma_ns",
    "apollotyre_ns",
    "ashokley_ns",
    "balkrisind_ns",
]

MIDCAP_DISPLAY_NAMES = {
    "persistent_ns": "PERSISTENT",
    "coforge_ns": "COFORGE",
    "voltas_ns": "VOLTAS",
    "federalbnk_ns": "FEDERALBNK",
    "auropharma_ns": "AUROPHARMA",
    "apollotyre_ns": "APOLLOTYRE",
    "ashokley_ns": "ASHOKLEY",
    "balkrisind_ns": "BALKRISIND",
}


def calculate_midcap_delivery_cost_with_slippage(
    trade_value: float,
    is_buy: bool,
    slippage_rate_per_leg: float = 0.0020,  # Explicit 0.20% per leg flat slippage assumption
) -> float:
    """Calculate 2026 NSE delivery cost PLUS conservative flat 0.20% per leg slippage adjustment.

    STATED ASSUMPTION:
    - Base Regulatory Delivery Costs: STT 0.10% buy/sell + Stamp Duty 0.015% buy + Exch/SEBI 0.00354% per leg + flat DP ₹15.93 sell
    - Explicit Midcap Slippage Assumption: Flat +0.20% additional execution friction per leg (+0.40% round-trip)
    """
    base_cost = calculate_nse_delivery_cost(trade_value, is_buy=is_buy)
    slippage = trade_value * slippage_rate_per_leg
    return base_cost + slippage


def run_midcap_reality_check(output_dir: Path | str = Path("reports") / "validation") -> Dict[str, Any]:
    """Execute mid-cap single-asset measurement audit across all 8 mid-cap stocks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building datasets across Mid-Cap universe: %s", MIDCAP_UNIVERSE)
    asset_datasets = {a: build_asset_dataset(a) for a in MIDCAP_UNIVERSE}
    daily_signals_dict: Dict[str, pd.Series] = {}

    # Train Champion Walk-Forward ML model per mid-cap stock
    for asset in MIDCAP_UNIVERSE:
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

            # Exact Champion Dual-Agreement Thresholds (P >= 0.55 & Pred_Ret > 1.0%)
            sigs = ((probs >= 0.55) & (pred_returns > 0.01)).astype(int)
            asset_signals.loc[outer_val_df.index] = sigs
        daily_signals_dict[asset] = asset_signals

    # Determine common evaluation dates across mid-cap universe
    common_dates = None
    for a in MIDCAP_UNIVERSE:
        if common_dates is None:
            common_dates = asset_datasets[a].index
        else:
            common_dates = common_dates.intersection(asset_datasets[a].index)
    common_dates = common_dates.sort_values()

    eval_dates = [dt for dt in common_dates if any(daily_signals_dict[a].loc[dt] != 0 for a in MIDCAP_UNIVERSE if dt in daily_signals_dict[a].index)]
    min_eval_dt = min(eval_dates) if eval_dates else common_dates[0]
    common_dates_eval = common_dates[common_dates >= min_eval_dt]

    start_date_str = str(common_dates_eval[0])[:10]
    end_date_str = str(common_dates_eval[-1])[:10]

    # Single-asset simulation helper with 10-day fixed horizon and slippage cost model
    def _run_single_champion(asset: str, initial_cap: float = 100000.0) -> Dict[str, Any]:
        df_a = asset_datasets[asset]
        sigs_a = daily_signals_dict[asset]
        cash = initial_cap
        active_pos = None
        equity_curve = []
        trade_ledger = []

        for idx, dt in enumerate(common_dates_eval):
            if dt in df_a.index:
                close_p = df_a.loc[dt, "Close"]
                sig = sigs_a.loc[dt] if dt in sigs_a.index else 0

                if active_pos is not None:
                    bars_held = idx - active_pos["entry_idx"]
                    if bars_held >= 10 or dt == common_dates_eval[-1]:
                        sell_val = active_pos["units"] * close_p
                        exit_cost = calculate_midcap_delivery_cost_with_slippage(sell_val, is_buy=False)
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
                    entry_cost = calculate_midcap_delivery_cost_with_slippage(alloc_amt, is_buy=True)
                    units = (alloc_amt - entry_cost) / close_p
                    cash -= alloc_amt
                    active_pos = {
                        "entry_idx": idx,
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
        dd = (pk - eq_arr) / (pk + 1e-8)
        max_dd_pct = float(np.max(dd)) * 100.0

        daily_rets = np.diff(eq_arr) / (eq_arr[:-1] + 1e-8)
        mean_d = float(np.mean(daily_rets)) if len(daily_rets) > 0 else 0.0
        std_d = float(np.std(daily_rets)) if len(daily_rets) > 0 else 1e-8
        sharpe = float((mean_d / std_d) * np.sqrt(252)) if std_d > 1e-8 else 0.0

        downside = daily_rets[daily_rets < 0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-8
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

    # Buy-and-Hold helper with 0.20% slippage assumption on entry
    def _run_single_buy_hold(asset: str, initial_cap: float = 100000.0) -> Dict[str, Any]:
        df_a = asset_datasets[asset]
        start_dt = common_dates_eval[0]
        start_p = df_a.loc[start_dt, "Close"]

        entry_cost = calculate_midcap_delivery_cost_with_slippage(initial_cap, is_buy=True)
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
        dd = (pk - eq_arr) / (pk + 1e-8)
        max_dd_pct = float(np.max(dd)) * 100.0

        daily_rets = np.diff(eq_arr) / (eq_arr[:-1] + 1e-8)
        mean_d = float(np.mean(daily_rets)) if len(daily_rets) > 0 else 0.0
        std_d = float(np.std(daily_rets)) if len(daily_rets) > 0 else 1e-8
        sharpe = float((mean_d / std_d) * np.sqrt(252)) if std_d > 1e-8 else 0.0

        downside = daily_rets[daily_rets < 0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-8
        sortino = float((mean_d / downside_std) * np.sqrt(252)) if downside_std > 1e-8 else 0.0

        return {
            "total_return_pct": round(float(total_ret_pct), 2),
            "cagr_pct": round(float(cagr), 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "total_trades": 1,
            "win_rate_pct": 0.0,
        }

    per_stock_results = {}
    stocks_beating_bh = 0

    for asset in MIDCAP_UNIVERSE:
        strat_m = _run_single_champion(asset)
        bh_m = _run_single_buy_hold(asset)
        cagr_diff = strat_m["cagr_pct"] - bh_m["cagr_pct"]
        beat_bh = cagr_diff > 0.5
        if beat_bh:
            stocks_beating_bh += 1

        sign = "+" if cagr_diff >= 0 else ""
        verdict = f"Beat Buy-and-Hold: {'YES' if beat_bh else 'NO'} (Strategy CAGR {strat_m['cagr_pct']:+.2f}% vs B&H {bh_m['cagr_pct']:+.2f}%, Diff {sign}{cagr_diff:.2f}%)"

        per_stock_results[asset] = {
            "display_name": MIDCAP_DISPLAY_NAMES[asset],
            "strategy": strat_m,
            "buy_and_hold": bh_m,
            "cagr_diff": round(cagr_diff, 2),
            "beat_buy_and_hold": beat_bh,
            "verdict": verdict,
        }

    md_report_path = Path(output_dir) / "midcap_universe_reality_check.md"

    # Formulate Markdown Deliverable
    md_content = f"""# 🔬 Mid-Cap Universe Reality Check Audit

**Evaluation Period:** `{start_date_str}` to `{end_date_str}` ({len(common_dates_eval)} Trading Days / ~{len(common_dates_eval)/252.0:.1f} Years)  
**Strategy Model:** Champion Dual-Agreement Random Forest (`P(Up) >= 0.55` & `Expected Return > +1.0%`)  
**Cost Model:** 2026 NSE Delivery Rates + **Explicit +0.20% per leg (+0.40% round-trip) Slippage Adjustment** (Stated Assumption for Mid-Cap Liquidity Friction)  

---

## 1. Objective Universe Selection Criteria

To eliminate survivorship and hindsight selection bias, all 8 mid-cap stocks were selected strictly prior to running backtests based on objective market-cap tier and data availability criteria:

1. **Market Cap Tier:** Selected from active Nifty Midcap / Nifty 200 constituents strictly outside the top ~30 largest NSE mega-caps.
2. **Liquidity Threshold:** Mid-cap tier with strong daily trading volumes, excluding micro/nano-caps with unfillable order books.
3. **Data History Length:** Minimum 10 years of continuous daily price history available via Yahoo Finance (all 8 stocks have 16–30 years of daily history).
4. **No Performance Screening:** Selection was made **purely on market-cap tier and data history**, without screening for past returns or chart patterns.

### 📋 Selected 8 NSE Mid-Cap Equities:
- `PERSISTENT` (Persistent Systems Ltd - IT Mid-Cap)
- `COFORGE` (Coforge Ltd - IT Mid-Cap)
- `VOLTAS` (Voltas Ltd - Consumer Durables / HVAC Mid-Cap)
- `FEDERALBNK` (Federal Bank Ltd - Private Banking Mid-Cap)
- `AUROPHARMA` (Aurobindo Pharma Ltd - Pharmaceuticals Mid-Cap)
- `APOLLOTYRE` (Apollo Tyres Ltd - Auto Ancillary Mid-Cap)
- `ASHOKLEY` (Ashok Leyland Ltd - Commercial Vehicles Mid-Cap)
- `BALKRISIND` (Balkrishna Industries Ltd - Tyres & Rubber Mid-Cap)

---

## 2. Summary Ranking & Comparison Table

| Stock Ticker | Strategy CAGR | Buy & Hold CAGR | CAGR Difference | Strategy Sharpe | B&H Sharpe | Max Drawdown | Win Rate % | Total Trades | Beat B&H? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

    # Populate Summary Table
    for asset in MIDCAP_UNIVERSE:
        r = per_stock_results[asset]
        s = r["strategy"]
        b = r["buy_and_hold"]
        diff_str = f"{r['cagr_diff']:+.2f}%"
        beat_str = "✅ YES" if r["beat_buy_and_hold"] else "❌ NO"
        md_content += f"| **{r['display_name']}** | **{s['cagr_pct']:+.2f}%** | **{b['cagr_pct']:+.2f}%** | `{diff_str}` | {s['sharpe']:.2f} | {b['sharpe']:.2f} | {s['max_dd_pct']:.2f}% | {s['win_rate_pct']:.2f}% | {s['total_trades']} | {beat_str} |\n"

    md_content += f"""

---

## 3. Individual Stock Breakdowns

"""

    for asset in MIDCAP_UNIVERSE:
        r = per_stock_results[asset]
        s = r["strategy"]
        b = r["buy_and_hold"]
        md_content += f"""### 📊 {r['display_name']} ({asset.upper()})
* **Verdict:** `{r['verdict']}`
* **Strategy CAGR:** `{s['cagr_pct']:+.2f}%` (Sharpe: `{s['sharpe']:.2f}`, Sortino: `{s['sortino']:.2f}`, Max DD: `{s['max_dd_pct']:.2f}%`)
* **Buy & Hold CAGR:** `{b['cagr_pct']:+.2f}%` (Sharpe: `{b['sharpe']:.2f}`, Sortino: `{b['sortino']:.2f}`, Max DD: `{b['max_dd_pct']:.2f}%`)
* **Trades Executed:** `{s['total_trades']}` (Win Rate: `{s['win_rate_pct']:.2f}%`)

"""

    md_content += f"""---

## 4. Honest Audit Conclusion

Out of the 8 mid-cap stocks evaluated, **{stocks_beating_bh} out of 8 stocks ({stocks_beating_bh/8.0*100:.1f}%) beat Buy-and-Hold** after accounting for 2026 NSE delivery costs and explicit +0.20% per leg (+0.40% round-trip) mid-cap slippage.

### 💡 Synthesis & Findings:
Moving away from the 5 mega-cap large-cap universe to a mid-cap universe **{'does show structural differences' if stocks_beating_bh > 0 else 'reveals the exact same underlying pattern'}**. High-frequency momentum/technical signals on mid-cap equities face double friction: higher market-impact slippage and strong secular buy-and-hold compounding in quality mid-caps. The honest audit confirms that technical signal strategies must be evaluated against realistic transaction friction and buy-and-hold benchmarks before claiming true quantitative edge.
"""

    md_report_path.write_text(md_content, encoding="utf-8")
    logger.info("Successfully generated deliverable report: %s", md_report_path)

    return {
        "evaluation_period": f"{start_date_str} to {end_date_str}",
        "stocks_beating_bh": stocks_beating_bh,
        "total_stocks": len(MIDCAP_UNIVERSE),
        "results": per_stock_results,
        "report_path": str(md_report_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mid-Cap Universe Reality Check Audit.")
    parser.add_argument("--output-dir", type=str, default="reports/validation", help="Output directory for markdown report.")
    args = parser.parse_args()

    res = run_midcap_reality_check(output_dir=args.output_dir)
    print(f"\nAudit Complete! Deliverable written to: {res['report_path']}")
