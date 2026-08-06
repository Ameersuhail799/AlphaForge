"""
Trend Feature Generator

Generates trend-based technical indicators.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrendFeatureGenerator:
    """
    Generates trend features.
    """

    REQUIRED_COLUMNS = ["Close"]

    def generate(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate trend features.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """

        logger.info("Generating trend features...")

        self._validate(df)

        df = df.copy()

        close = df["Close"]

        df["SMA_20"] = close.rolling(window=20).mean()

        df["SMA_50"] = close.rolling(window=50).mean()

        df["EMA_20"] = close.ewm(
            span=20,
            adjust=False,
        ).mean()

        df["EMA_50"] = close.ewm(
            span=50,
            adjust=False,
        ).mean()

        logger.info(
            "Trend features generated successfully."
        )

        return df

    def _validate(
        self,
        df: pd.DataFrame,
    ) -> None:

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )