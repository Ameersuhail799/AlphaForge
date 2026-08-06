"""Chronological splitting for AlphaForge datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChronologicalSplit:
    """Contain chronological feature, target, and split-date partitions."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_start: Hashable
    train_end: Hashable
    validation_start: Hashable
    validation_end: Hashable
    test_start: Hashable
    test_end: Hashable


class ChronologicalSplitter:
    """Split feature data into ordered train, validation, and test sets."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        """Initialize chronological split ratios.

        Args:
            train_ratio: Proportion of rows allocated to training.
            validation_ratio: Proportion of rows allocated to validation.
            test_ratio: Proportion of rows allocated to testing.

        Raises:
            ValueError: If ratios are invalid or do not total one.
        """

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

        self._validate_ratios()

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> ChronologicalSplit:
        """Split features and targets without shuffling.

        Args:
            X: Feature matrix ordered chronologically.
            y: Target series aligned with the feature matrix.

        Returns:
            Chronological train, validation, and test partitions.

        Raises:
            ValueError: If inputs are misaligned or contain too few rows.
        """

        logger.info("Creating chronological dataset splits...")

        self._validate_inputs(X, y)

        total_rows = len(X)
        train_end = int(total_rows * self.train_ratio)
        validation_end = train_end + int(
            total_rows * self.validation_ratio
        )

        X_train = X.iloc[:train_end].copy()
        y_train = y.iloc[:train_end].copy()
        X_valid = X.iloc[train_end:validation_end].copy()
        y_valid = y.iloc[train_end:validation_end].copy()
        X_test = X.iloc[validation_end:].copy()
        y_test = y.iloc[validation_end:].copy()

        result = ChronologicalSplit(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            train_start=X_train.index[0],
            train_end=X_train.index[-1],
            validation_start=X_valid.index[0],
            validation_end=X_valid.index[-1],
            test_start=X_test.index[0],
            test_end=X_test.index[-1],
        )

        logger.info(
            "Chronological splits created: train=%d, validation=%d, test=%d.",
            len(X_train),
            len(X_valid),
            len(X_test),
        )

        return result

    def _validate_ratios(self) -> None:
        """Validate configured split ratios.

        Raises:
            ValueError: If ratios are not positive or do not total one.
        """

        ratios = [
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        ]

        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("Split ratios must be greater than zero.")

        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1.0.")

    def _validate_inputs(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Validate split inputs.

        Args:
            X: Feature matrix.
            y: Target series.

        Raises:
            ValueError: If inputs cannot form three non-empty partitions.
        """

        if len(X) != len(y) or not X.index.equals(y.index):
            raise ValueError("Features and targets must have matching indexes.")

        if len(X) < 3:
            raise ValueError("At least three rows are required for splitting.")
