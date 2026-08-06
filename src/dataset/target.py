"""Target generation for AlphaForge datasets."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TargetBuilder:
    """Build supervised learning targets from market data."""

    DEFAULT_TARGET = "NEXT_DAY_DIRECTION"
    REQUIRED_COLUMNS = ["Close"]

    def __init__(self) -> None:
        """Initialize the available target-generation strategies."""

        self._target_builders: dict[
            str,
            Callable[[pd.DataFrame], pd.Series],
        ] = {
            self.DEFAULT_TARGET: self._build_next_day_direction,
        }

    def build(
        self,
        df: pd.DataFrame,
        target_name: str = DEFAULT_TARGET,
    ) -> pd.DataFrame:
        """Add a target column and remove rows without a target value.

        Args:
            df: Feature-engineered market data.
            target_name: Name of the configured target to create.

        Returns:
            DataFrame containing the requested target column.

        Raises:
            ValueError: If required columns or the target strategy are missing.
        """

        logger.info("Building target '%s'...", target_name)

        self._validate(df)

        target_builder = self._target_builders.get(target_name)

        if target_builder is None:
            raise ValueError(f"Unsupported target: {target_name}")

        result = df.copy()
        result[target_name] = target_builder(result)
        result = result.dropna(subset=[target_name])
        result[target_name] = result[target_name].astype(int)

        logger.info("Target '%s' built successfully.", target_name)

        return result

    def _build_next_day_direction(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """Build the next-day direction classification target.

        Args:
            df: Feature-engineered market data.

        Returns:
            One when tomorrow's close exceeds today's close, otherwise zero.
        """

        return (df["Close"].shift(-1) > df["Close"]).where(
            df["Close"].shift(-1).notna()
        )

    def _validate(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Validate required input columns.

        Args:
            df: Feature-engineered market data.

        Raises:
            ValueError: If required input columns are missing.
        """

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(f"Missing columns: {missing}")
