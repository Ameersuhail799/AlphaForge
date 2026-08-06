"""Dataset construction for AlphaForge machine learning workflows."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.settings import DEFAULT_SYMBOL
from src.dataset.bundle import DatasetBundle
from src.dataset.scaler import FeatureScaler
from src.dataset.splitter import ChronologicalSplitter
from src.dataset.target import TargetBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """Build leakage-safe machine learning datasets from engineered features."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        scale: bool = False,
    ) -> None:
        """Initialize dataset construction dependencies.

        Args:
            train_ratio: Proportion allocated to training.
            validation_ratio: Proportion allocated to validation.
            test_ratio: Proportion allocated to testing.
            scale: Whether to standardize features using training data only.
        """

        self.target_builder = TargetBuilder()
        self.splitter = ChronologicalSplitter(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        self.scaler = FeatureScaler(scale=scale)

    def build(
        self,
        df: pd.DataFrame,
        symbol: str = DEFAULT_SYMBOL,
        target_name: str = TargetBuilder.DEFAULT_TARGET,
    ) -> DatasetBundle:
        """Build a chronological, optionally scaled dataset bundle.

        Args:
            df: Feature-engineered market data.
            symbol: Market symbol represented by the dataset.
            target_name: Name of the target strategy to create.

        Returns:
            Leakage-safe training, validation, and test datasets.

        Raises:
            ValueError: If no usable rows remain after removing missing values.
        """

        logger.info("Building dataset for %s...", symbol)

        clean_df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna().copy()

        if clean_df.empty:
            raise ValueError("No rows remain after removing missing values.")

        target_df = self.target_builder.build(clean_df, target_name)
        X = target_df.drop(columns=[target_name])
        y = target_df[target_name]

        split = self.splitter.split(X, y)

        X_train = self.scaler.fit_transform_train(split.X_train)
        X_valid = self.scaler.transform(split.X_valid)
        X_test = self.scaler.transform(split.X_test)

        metadata = {
            "symbol": symbol,
            "rows": len(target_df),
            "feature_count": len(X.columns),
            "target": target_name,
            "train_ratio": self.splitter.train_ratio,
            "validation_ratio": self.splitter.validation_ratio,
            "test_ratio": self.splitter.test_ratio,
            "scaled": self.scaler.scale,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        bundle = DatasetBundle(
            X_train=X_train,
            y_train=split.y_train,
            X_valid=X_valid,
            y_valid=split.y_valid,
            X_test=X_test,
            y_test=split.y_test,
            feature_names=X.columns.tolist(),
            target_name=target_name,
            train_start=split.train_start,
            train_end=split.train_end,
            validation_start=split.validation_start,
            validation_end=split.validation_end,
            test_start=split.test_start,
            test_end=split.test_end,
            metadata=metadata,
        )

        logger.info("Dataset built successfully for %s.", symbol)

        return bundle
