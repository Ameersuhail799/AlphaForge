"""
Yahoo Finance Provider

This module is responsible for downloading historical market data
from Yahoo Finance.

Responsibilities:
- Download historical price data
- Download dividends and stock splits
- Handle download errors
- Return a pandas DataFrame

This module NEVER:
- Saves data
- Validates data
- Performs feature engineering
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def is_nse_market_open(now_dt: Optional[datetime] = None) -> bool:
    """Check if National Stock Exchange (NSE) is currently in active trading hours.

    NSE Trading Hours: Monday - Friday, 9:15 AM - 3:30 PM IST (UTC+05:30).
    """
    ist = timezone(timedelta(hours=5, minutes=30))
    dt = now_dt.astimezone(ist) if now_dt else datetime.now(ist)
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    time_num = dt.hour * 60 + dt.minute
    market_open_num = 9 * 60 + 15   # 09:15 IST
    market_close_num = 15 * 60 + 30 # 15:30 IST
    return market_open_num <= time_num <= market_close_num


_LIVE_PRICE_CACHE: Dict[str, tuple[Dict[str, Any], float]] = {}


class YahooProvider:
    """Yahoo Finance data provider."""

    def __init__(self):
        logger.info("YahooProvider initialized.")

    def download(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download historical market data.

        Args:
            symbol: Stock ticker.
            start_date: Start date.
            end_date: End date.
            interval: Data interval.

        Returns:
            Pandas DataFrame containing market data.
        """
        logger.info(f"Downloading data for {symbol}")

        try:
            data = yf.download(
                tickers=symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )

            if data.empty:
                raise ValueError("Downloaded dataset is empty.")

            logger.info(f"Successfully downloaded {len(data)} rows.")
            return data

        except Exception as error:
            logger.exception("Yahoo download failed.")
            raise RuntimeError(f"Failed downloading {symbol}") from error

    def get_live_price(self, symbol: str, fallback_close: float = 0.0) -> Dict[str, Any]:
        """Fetch current/latest market price for symbol during NSE market hours or return last close."""
        import time

        cache_key = symbol.lower()
        now_ts = time.time()
        if cache_key in _LIVE_PRICE_CACHE:
            cached_data, cached_time = _LIVE_PRICE_CACHE[cache_key]
            if now_ts - cached_time < 30.0:  # 30-second TTL cache
                return cached_data

        ticker_sym = symbol.replace("_", ".").upper()
        if not ticker_sym.endswith(".NS"):
            ticker_sym = f"{ticker_sym}.NS"

        is_open = is_nse_market_open()
        ist = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(ist)

        price = fallback_close
        prev_close = fallback_close

        try:
            t = yf.Ticker(ticker_sym)
            info = getattr(t, "fast_info", {})
            last_p = info.get("lastPrice") or info.get("regularMarketPrice")
            prev_c = info.get("previousClose") or info.get("regularMarketPreviousClose")

            if last_p is not None and not pd.isna(last_p) and float(last_p) > 0:
                price = float(last_p)
            if prev_c is not None and not pd.isna(prev_c) and float(prev_c) > 0:
                prev_close = float(prev_c)
        except Exception as err:
            logger.warning("yfinance live price fetch failed for %s: %s", symbol, str(err))

        if prev_close <= 0:
            prev_close = price if price > 0 else 1.0

        change_val = price - prev_close
        change_pct = (change_val / prev_close) * 100.0 if prev_close > 0 else 0.0

        status_text = f"Market Open ({now_ist.strftime('%H:%M IST')})" if is_open else f"Market Closed (As of {now_ist.strftime('%d %b')})"

        res_dict = {
            "symbol": symbol,
            "ticker": ticker_sym,
            "current_price": round(price, 2),
            "previous_close": round(prev_close, 2),
            "change_val": round(change_val, 2),
            "change_pct": round(change_pct, 2),
            "is_market_open": is_open,
            "market_status_text": status_text,
            "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
            "fetch_time_formatted": now_ist.strftime("%H:%M:%S IST"),
        }
        _LIVE_PRICE_CACHE[cache_key] = (res_dict, now_ts)
        return res_dict