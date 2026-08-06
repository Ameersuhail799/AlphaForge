"""
Price Feature Generator

Generates price action and candlestick-based technical indicators.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PriceFeatureGenerator:
    """Generates price action features from OHLC market data."""

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close"]

    def generate(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate price action features.

        Args:
            df: Input market data containing OHLC price columns.

        Returns:
            DataFrame containing the original data and price features.

        Raises:
            ValueError: If required input columns are missing.
        """

        logger.info("Generating price features...")

        self._validate(df)

        df = df.copy()

        open_price = df["Open"]
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        previous_close = close.shift(1)

        df["GAP_PCT"] = (
            (open_price - previous_close) / previous_close
        ) * 100
        df["OPEN_CLOSE_PCT"] = (
            (close - open_price) / open_price
        ) * 100
        df["HIGH_LOW_PCT"] = ((high - low) / close) * 100
        df["BODY_SIZE"] = (close - open_price).abs()
        df["UPPER_WICK"] = high - pd.concat(
            [open_price, close],
            axis=1,
        ).max(axis=1)
        df["LOWER_WICK"] = pd.concat(
            [open_price, close],
            axis=1,
        ).min(axis=1) - low

        logger.info("Price features generated successfully.")

        return df

    def _validate(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Validate that required OHLC columns are available.

        Args:
            df: Input market data.

        Raises:
            ValueError: If required input columns are missing.
        """

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )
