"""
Volume Feature Generator

Generates volume-based technical indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VolumeFeatureGenerator:
    """Generates volume-based features."""

    REQUIRED_COLUMNS = ["Close", "Volume"]

    def generate(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate volume features.

        Args:
            df: Input market data containing close prices and volume.

        Returns:
            DataFrame containing the original data and volume features.

        Raises:
            ValueError: If required input columns are missing.
        """

        logger.info("Generating volume features...")

        self._validate(df)

        df = df.copy()

        volume = df["Volume"]
        close = df["Close"]

        df["VOLUME_SMA_20"] = volume.rolling(window=20).mean()
        df["VOLUME_EMA_20"] = volume.ewm(
            span=20,
            adjust=False,
        ).mean()
        df["VOLUME_RATIO"] = volume / df["VOLUME_SMA_20"]
        df["VOLUME_CHANGE_PCT"] = volume.pct_change()

        direction = np.sign(close.diff()).fillna(0)
        df["OBV"] = (direction * volume).cumsum()

        logger.info("Volume features generated successfully.")

        return df

    def _validate(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Validate that required market data columns are available.

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
