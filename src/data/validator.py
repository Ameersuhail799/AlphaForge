"""
Data Validation Module

Validates downloaded market data before it enters
the AlphaForge pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ValidationReport:
    """
    Stores validation results.
    """

    passed: bool
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    duplicate_dates: int
    future_dates: int
    negative_prices: int
    negative_volume: int
    message: str


class DataValidator:
    """
    Validate market datasets.
    """

    REQUIRED_COLUMNS = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    def validate(self, df: pd.DataFrame) -> ValidationReport:

        logger.info("Starting dataset validation.")

        if df.empty:
            return ValidationReport(
                False,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                "Dataset is empty.",
            )

        # Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Flattening MultiIndex columns.")
            df.columns = df.columns.get_level_values(0)

        missing_columns = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            return ValidationReport(
                False,
                len(df),
                len(df.columns),
                0,
                0,
                0,
                0,
                0,
                0,
                f"Missing columns: {missing_columns}",
            )

        missing_values = int(df.isna().sum().sum())

# Duplicate rows are not considered an error for
# financial time-series because different trading
# days may legitimately contain identical values.
        duplicate_rows = 0
        duplicate_dates = int(df.index.duplicated().sum())

        future_dates = int(
            (df.index > pd.Timestamp.today()).sum()
        )

        negative_prices = int(
            (
                (df["Open"] < 0)
                | (df["High"] < 0)
                | (df["Low"] < 0)
                | (df["Close"] < 0)
            ).sum()
        )

        negative_volume = int(
            (df["Volume"] < 0).sum()
        )

        passed = (
            missing_values == 0
            and duplicate_dates == 0
            and future_dates == 0
            and negative_prices == 0
            and negative_volume == 0
        )

        report = ValidationReport(
            passed=passed,
            rows=len(df),
            columns=len(df.columns),
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            duplicate_dates=duplicate_dates,
            future_dates=future_dates,
            negative_prices=negative_prices,
            negative_volume=negative_volume,
            message="Validation successful."
            if passed
            else "Validation failed.",
        )

        logger.info(report)

        return report