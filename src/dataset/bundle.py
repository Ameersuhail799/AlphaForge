"""Dataset bundle definitions for machine learning workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import pandas as pd


@dataclass
class DatasetBundle:
    """Contain chronological train, validation, and test datasets.

    Attributes:
        X_train: Training feature matrix.
        y_train: Training target values.
        X_valid: Validation feature matrix.
        y_valid: Validation target values.
        X_test: Test feature matrix.
        y_test: Test target values.
        feature_names: Ordered feature column names.
        target_name: Name of the target column.
        train_start: First index value in the training split.
        train_end: Last index value in the training split.
        validation_start: First index value in the validation split.
        validation_end: Last index value in the validation split.
        test_start: First index value in the test split.
        test_end: Last index value in the test split.
        metadata: Dataset construction metadata.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]
    target_name: str
    train_start: Hashable
    train_end: Hashable
    validation_start: Hashable
    validation_end: Hashable
    test_start: Hashable
    test_end: Hashable
    metadata: dict[str, object]
