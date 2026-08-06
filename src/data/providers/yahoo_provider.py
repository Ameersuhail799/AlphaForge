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

import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)


class YahooProvider:
    """
    Yahoo Finance data provider.
    """

    def __init__(self):
        logger.info("YahooProvider initialized.")

    def download(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download historical market data.

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

            logger.info(
                f"Successfully downloaded {len(data)} rows."
            )

            return data

        except Exception as error:

            logger.exception("Yahoo download failed.")

            raise RuntimeError(
                f"Failed downloading {symbol}"
            ) from error