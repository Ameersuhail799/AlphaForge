"""
Momentum Feature Generator
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MomentumFeatureGenerator:
    """
    Generates momentum-based features.
    """

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Generating momentum features...")

        df = df.copy()

        close = df["Close"]

        # Daily Return
        df["DAILY_RETURN"] = close.pct_change()

        # Price Change %
        df["PRICE_CHANGE_PCT"] = (
            (close - close.shift(1)) / close.shift(1)
        ) * 100

        # Momentum
        df["MOMENTUM_10"] = close - close.shift(10)
        df["MOMENTUM_20"] = close - close.shift(20)

        # Rate of Change
        df["ROC_12"] = (
            (close - close.shift(12))
            / close.shift(12)
        ) * 100

        # RSI (14)
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss

        df["RSI_14"] = 100 - (100 / (1 + rs))

        logger.info("Momentum features generated successfully.")

        return df