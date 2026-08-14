"""Historical Context Research Module for AlphaForge.

Analyzes historical market data to find all prior trading days matching
today's exact 3-way regime bucket (RSI zone, Trend regime, Volatility regime)
and computes 10-day forward return distribution statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.research.benchmark_and_cost_reality_check import (
    ASSET_UNIVERSE,
    build_asset_dataset,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_historical_context(symbol: str = "tcs_ns") -> Dict[str, Any]:
    """Calculate historical context distribution for today's regime bucket.

    Parameters
    ----------
    symbol : str
        Asset symbol identifier (e.g. 'tcs_ns', 'infy_ns').

    Returns
    -------
    Dict[str, Any]
        Structured dictionary containing regime description, sample size N,
        and 10-day forward return metrics (min, p25, median, p75, max, pct_positive)
        or an insufficient sample flag if N < 20.
    """
    symbol = symbol.lower().strip()
    if symbol not in ASSET_UNIVERSE:
        symbol = "tcs_ns"

    df = build_asset_dataset(symbol)
    asset_display_name = symbol.replace("_ns", "").upper()

    if len(df) < 50:
        return {
            "symbol": symbol,
            "display_name": asset_display_name,
            "insufficient_sample": True,
            "sample_count": 0,
            "regime_description": "Unknown",
            "message": "Not enough historical market data to calculate regime statistics.",
            "metrics": None,
        }

    close = df["Close"]
    sma20 = df.get("SMA_20", close.rolling(20).mean())
    sma50 = df.get("SMA_50", close.rolling(50).mean())
    rsi = df.get("RSI_14", pd.Series(50.0, index=df.index))
    hist_vol = df.get("HIST_VOL_20", pd.Series(0.20, index=df.index))

    vol_med60 = hist_vol.rolling(60, min_periods=20).median()

    # 1. RSI Zone Definition
    rsi_zones = np.where(
        rsi < 30,
        "Oversold (RSI < 30)",
        np.where(rsi > 70, "Overbought (RSI > 70)", "Neutral RSI (30-70)"),
    )

    # 2. Trend Regime Definition
    if "BULLISH_TREND_REGIME" in df.columns and "BEARISH_TREND_REGIME" in df.columns:
        trend_regimes = np.where(
            df["BULLISH_TREND_REGIME"] == 1,
            "Bullish Trend",
            np.where(df["BEARISH_TREND_REGIME"] == 1, "Bearish Trend", "Neutral Trend"),
        )
    else:
        is_bullish = (close > sma50) & (sma20 > sma50)
        is_bearish = (close < sma50) & (sma20 < sma50)
        trend_regimes = np.where(
            is_bullish,
            "Bullish Trend",
            np.where(is_bearish, "Bearish Trend", "Neutral Trend"),
        )

    # 3. Volatility Regime Definition
    if "HIGH_VOLATILITY_REGIME" in df.columns and "LOW_VOLATILITY_REGIME" in df.columns:
        vol_regimes = np.where(
            df["HIGH_VOLATILITY_REGIME"] == 1,
            "High Volatility",
            np.where(df["LOW_VOLATILITY_REGIME"] == 1, "Low Volatility", "Neutral Volatility"),
        )
    else:
        is_high_vol = hist_vol > (vol_med60 * 1.15)
        is_low_vol = hist_vol < (vol_med60 * 0.85)
        vol_regimes = np.where(
            is_high_vol,
            "High Volatility",
            np.where(is_low_vol, "Low Volatility", "Neutral Volatility"),
        )

    df_reg = df.copy()
    df_reg["RSI_ZONE"] = rsi_zones
    df_reg["TREND_REGIME"] = trend_regimes
    df_reg["VOL_REGIME"] = vol_regimes

    # Identify today's bucket
    today_row = df_reg.iloc[-1]
    today_rsi_zone = str(today_row["RSI_ZONE"])
    today_trend = str(today_row["TREND_REGIME"])
    today_vol = str(today_row["VOL_REGIME"])

    regime_description = f"{today_rsi_zone}, {today_trend}, {today_vol}"

    # Filter historical matching days (excluding recent 15 trading days)
    hist_df = df_reg.iloc[:-15]
    match_mask = (
        (hist_df["RSI_ZONE"] == today_rsi_zone)
        & (hist_df["TREND_REGIME"] == today_trend)
        & (hist_df["VOL_REGIME"] == today_vol)
    )
    matching_df = hist_df[match_mask].dropna(subset=["REALIZED_RET_10D"])

    sample_count = len(matching_df)

    # Rule 4: If N < 20, return insufficient_sample: True
    if sample_count < 20:
        return {
            "symbol": symbol,
            "display_name": asset_display_name,
            "insufficient_sample": True,
            "sample_count": sample_count,
            "regime_description": regime_description,
            "message": f"Not enough similar days in history (N={sample_count}) to draw a pattern.",
            "metrics": None,
        }

    rets = matching_df["REALIZED_RET_10D"].values * 100.0

    pct_positive = float(np.mean(rets > 0) * 100.0)
    min_ret = float(np.min(rets))
    p25_ret = float(np.percentile(rets, 25))
    median_ret = float(np.median(rets))
    p75_ret = float(np.percentile(rets, 75))
    max_ret = float(np.max(rets))

    return {
        "symbol": symbol,
        "display_name": asset_display_name,
        "insufficient_sample": False,
        "sample_count": sample_count,
        "regime_description": regime_description,
        "message": f"This exact combination occurred {sample_count} times before for {asset_display_name}.",
        "metrics": {
            "pct_positive": round(pct_positive, 2),
            "min_ret_pct": round(min_ret, 2),
            "p25_ret_pct": round(p25_ret, 2),
            "median_ret_pct": round(median_ret, 2),
            "p75_ret_pct": round(p75_ret, 2),
            "max_ret_pct": round(max_ret, 2),
        },
    }
