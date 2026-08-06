"""
Volatility Feature Generator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VolatilityFeatureGenerator:

    def generate(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info("Generating volatility features...")

        df = df.copy()

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        previous_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - previous_close).abs()
        tr3 = (low - previous_close).abs()

        df["TRUE_RANGE"] = np.maximum.reduce([tr1, tr2, tr3])

        df["ATR_14"] = (
            pd.Series(df["TRUE_RANGE"])
            .rolling(14)
            .mean()
        )

        returns = np.log(close / close.shift(1))

        df["HIST_VOL_20"] = (
            returns.rolling(20).std()
            * np.sqrt(252)
        )

        df["ROLLING_STD_20"] = (
            close.rolling(20).std()
        )

        df["DAILY_RANGE_PCT"] = (
            (high - low) / close
        ) * 100

        logger.info("Volatility features generated successfully.")

        return df