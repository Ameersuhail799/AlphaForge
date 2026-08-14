"""Strategy Tester Engine for AlphaForge.

Allows testing user-defined deterministic technical rule templates against
realistic 2026 NSE delivery transaction costs and a Buy-and-Hold benchmark.

Supported Templates:
1. rsi_threshold: Buy when RSI(period) crosses below buy_threshold.
2. sma_crossover: Buy when Fast SMA crosses above Slow SMA.
3. price_vs_sma: Buy when Close crosses above SMA(period).
4. volume_breakout: Buy when Volume > multiplier x 20-day SMA volume AND Close > Open.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.research.benchmark_and_cost_reality_check import (
    ASSET_UNIVERSE,
    build_asset_dataset,
    calculate_nse_delivery_cost,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def generate_template_signals(
    df: pd.DataFrame,
    template: str,
    params: Dict[str, Any],
) -> pd.Series:
    """Generate 1 (BUY) or 0 (HOLD/NONE) deterministic daily signals for a given template."""
    signals = pd.Series(0, index=df.index, dtype=int)

    close = df["Close"]
    open_p = df.get("Open", df["Close"])
    volume = df.get("Volume", pd.Series(1000, index=df.index))

    if template == "rsi_threshold":
        period = int(params.get("period", 14))
        buy_threshold = float(params.get("buy_threshold", 30))
        rsi = compute_rsi(close, period=period)
        # Buy when RSI is below threshold
        signals = (rsi < buy_threshold).astype(int)

    elif template == "sma_crossover":
        fast = int(params.get("fast", 20))
        slow = int(params.get("slow", 50))
        sma_fast = close.rolling(window=fast, min_periods=fast).mean()
        sma_slow = close.rolling(window=slow, min_periods=slow).mean()
        # Buy when fast SMA > slow SMA and fast SMA[t-1] <= slow SMA[t-1] (bullish crossover)
        crossover = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
        signals = crossover.astype(int)

    elif template == "price_vs_sma":
        period = int(params.get("period", 50))
        sma = close.rolling(window=period, min_periods=period).mean()
        # Buy when Close crosses above SMA
        cross_above = (close > sma) & (close.shift(1) <= sma.shift(1))
        signals = cross_above.astype(int)

    elif template == "volume_breakout":
        multiplier = float(params.get("multiplier", 2.0))
        vol_sma20 = volume.rolling(window=20, min_periods=20).mean()
        # Buy when volume > multiplier * 20-day SMA volume AND day closes up
        breakout = (volume > multiplier * vol_sma20) & (close > open_p)
        signals = breakout.astype(int)

    else:
        logger.warning("Unknown strategy template '%s'. Defaulting to 0 signals.", template)

    return signals.fillna(0).astype(int)


def run_strategy_backtest(
    asset_selection: str = "tcs_ns",
    template: str = "rsi_threshold",
    params: Optional[Dict[str, Any]] = None,
    exit_days: int = 10,
    initial_cap: float = 100000.0,
) -> Dict[str, Any]:
    """Simulate user strategy vs Buy-and-Hold benchmark with realistic 2026 NSE delivery costs.

    Parameters:
    - asset_selection: Single asset symbol ('tcs_ns', 'infy_ns', etc.) or 'all_pooled'.
    - template: One of ['rsi_threshold', 'sma_crossover', 'price_vs_sma', 'volume_breakout'].
    - params: Dictionary of parameter values for the selected template.
    - exit_days: User-adjustable exit horizon (1 to 30 trading days).
    - initial_cap: Starting capital (INR 1,00,000).
    """
    if params is None:
        params = {}

    exit_days = max(1, min(30, int(exit_days)))

    if asset_selection == "all_pooled":
        assets_to_run = list(ASSET_UNIVERSE)
    else:
        symbol = asset_selection.lower().strip()
        if symbol not in ASSET_UNIVERSE:
            symbol = "tcs_ns"
        assets_to_run = [symbol]

    # Load datasets
    datasets = {a: build_asset_dataset(a) for a in assets_to_run}

    # Find common evaluation dates
    common_dates = None
    for a in assets_to_run:
        if common_dates is None:
            common_dates = datasets[a].index
        else:
            common_dates = common_dates.intersection(datasets[a].index)
    common_dates = common_dates.sort_values()

    # Generate signals for each asset
    signals_dict = {
        a: generate_template_signals(datasets[a], template, params) for a in assets_to_run
    }

    # 1. Run User Strategy Simulation
    def _simulate_strategy() -> Tuple[Dict[str, Any], List[float]]:
        n_assets = len(assets_to_run)
        alloc_per_asset = initial_cap / n_assets
        cash = initial_cap
        active_positions: Dict[str, Optional[Dict[str, Any]]] = {a: None for a in assets_to_run}
        equity_curve: List[float] = []
        trade_ledger: List[Dict[str, Any]] = []

        for idx, dt in enumerate(common_dates):
            # Process exits first
            for a in assets_to_run:
                pos = active_positions[a]
                if pos is not None:
                    bars_held = idx - pos["entry_bar_idx"]
                    if bars_held >= exit_days or dt == common_dates[-1]:
                        df_a = datasets[a]
                        close_p = df_a.loc[dt, "Close"] if dt in df_a.index else pos["entry_price"]
                        sell_val = pos["units"] * close_p
                        exit_cost = calculate_nse_delivery_cost(sell_val, is_buy=False)
                        net_pnl = (sell_val - exit_cost) - pos["total_entry_outflow"]
                        cash += pos["alloc"] + net_pnl
                        trade_ledger.append({
                            "asset": a,
                            "net_pnl": net_pnl,
                            "net_return": net_pnl / pos["alloc"],
                            "is_win": (net_pnl > 0),
                        })
                        active_positions[a] = None

            # Process entries
            for a in assets_to_run:
                if active_positions[a] is None:
                    sig = signals_dict[a].loc[dt] if dt in signals_dict[a].index else 0
                    if sig == 1:
                        df_a = datasets[a]
                        close_p = df_a.loc[dt, "Close"] if dt in df_a.index else None
                        if close_p is not None and close_p > 0:
                            alloc_amt = min(cash, alloc_per_asset)
                            if alloc_amt > 100.0:
                                entry_cost = calculate_nse_delivery_cost(alloc_amt, is_buy=True)
                                units = (alloc_amt - entry_cost) / close_p
                                cash -= alloc_amt
                                active_positions[a] = {
                                    "entry_bar_idx": idx,
                                    "entry_date": dt,
                                    "entry_price": close_p,
                                    "alloc": alloc_amt,
                                    "units": units,
                                    "entry_cost": entry_cost,
                                    "total_entry_outflow": alloc_amt,
                                }

            # Compute portfolio equity for the day
            pos_val = 0.0
            for a in assets_to_run:
                pos = active_positions[a]
                if pos is not None:
                    df_a = datasets[a]
                    curr_p = df_a.loc[dt, "Close"] if dt in df_a.index else pos["entry_price"]
                    pos_val += pos["units"] * curr_p
            eq = cash + pos_val
            equity_curve.append(eq)

        metrics = _compute_metrics(equity_curve, trade_ledger, initial_cap, len(common_dates))
        return metrics, equity_curve

    # 2. Run Buy-and-Hold Benchmark Simulation
    def _simulate_buy_and_hold() -> Dict[str, Any]:
        n_assets = len(assets_to_run)
        alloc_per_asset = initial_cap / n_assets
        start_dt = common_dates[0]
        units_dict = {}

        for a in assets_to_run:
            df_a = datasets[a]
            start_p = df_a.loc[start_dt, "Close"] if start_dt in df_a.index else df_a["Close"].iloc[0]
            entry_cost = calculate_nse_delivery_cost(alloc_per_asset, is_buy=True)
            units = (alloc_per_asset - entry_cost) / start_p
            units_dict[a] = units

        equity_curve: List[float] = []
        for dt in common_dates:
            eq = 0.0
            for a in assets_to_run:
                df_a = datasets[a]
                curr_p = df_a.loc[dt, "Close"] if dt in df_a.index else df_a["Close"].iloc[0]
                eq += units_dict[a] * curr_p
            equity_curve.append(eq)

        return _compute_metrics(equity_curve, [], initial_cap, len(common_dates))

    strat_metrics, strat_eq = _simulate_strategy()
    bh_metrics = _simulate_buy_and_hold()

    # Formulate Plain Verdict Sentence
    cagr_diff = strat_metrics["cagr_pct"] - bh_metrics["cagr_pct"]
    if cagr_diff > 0.5:
        verdict = f"Beat Buy-and-Hold: YES (Strategy CAGR {strat_metrics['cagr_pct']:+.2f}% vs Buy-and-Hold {bh_metrics['cagr_pct']:+.2f}%, Outperformance {cagr_diff:+.2f}%)"
    elif cagr_diff < -0.5:
        verdict = f"Beat Buy-and-Hold: NO (Strategy CAGR {strat_metrics['cagr_pct']:+.2f}% vs Buy-and-Hold {bh_metrics['cagr_pct']:+.2f}%, Underperformance {cagr_diff:+.2f}%)"
    else:
        verdict = f"Beat Buy-and-Hold: CLOSE (Strategy CAGR {strat_metrics['cagr_pct']:+.2f}% vs Buy-and-Hold {bh_metrics['cagr_pct']:+.2f}%, Difference {cagr_diff:+.2f}%)"

    asset_display_name = "All 5 Equities Pooled" if asset_selection == "all_pooled" else asset_selection.replace("_ns", "").upper()

    return {
        "asset_selection": asset_selection,
        "asset_display_name": asset_display_name,
        "template": template,
        "params": params,
        "exit_days": exit_days,
        "verdict": verdict,
        "beat_buy_and_hold": (cagr_diff > 0.5),
        "evaluation_period": {
            "start_date": str(common_dates[0])[:10],
            "end_date": str(common_dates[-1])[:10],
            "trading_days": len(common_dates),
        },
        "strategy_metrics": strat_metrics,
        "benchmark_metrics": bh_metrics,
    }


def _compute_metrics(
    equity_curve: List[float],
    trade_ledger: List[Dict[str, Any]],
    initial_cap: float,
    n_days: int,
) -> Dict[str, Any]:
    """Compute standard financial evaluation metrics."""
    eq_arr = np.array(equity_curve)
    if len(eq_arr) == 0:
        return {
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd_pct": 0.0,
            "total_trades": 0,
            "win_rate_pct": 0.0,
        }

    total_ret_pct = float(((eq_arr[-1] - initial_cap) / initial_cap) * 100.0)
    n_years = max(n_days / 252.0, 0.1)
    cagr = float(((eq_arr[-1] / initial_cap) ** (1.0 / n_years) - 1.0) * 100.0)

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
        "total_return_pct": round(total_ret_pct, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
    }
