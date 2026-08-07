"""Automatic DatasetBundle feature subset creation for AlphaForge research."""

from __future__ import annotations

import pandas as pd

from src.dataset.bundle import DatasetBundle
from src.research.feature_optimizer import FeatureOptimizationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Create DatasetBundle feature subsets from optimized feature rankings."""

    def create_subsets(
        self,
        bundle: DatasetBundle,
        optimization: FeatureOptimizationResult,
    ) -> dict[int, DatasetBundle]:
        """Create top-ranked feature subsets without manual column edits.

        Args:
            bundle: Source dataset bundle.
            optimization: Aggregated feature ranking result.

        Returns:
            Dataset bundles keyed by requested feature count.
        """

        logger.info("Creating optimized feature subset datasets...")

        subsets = {
            count: self._create_subset(
                bundle,
                optimization.top_features[count],
            )
            for count in optimization.top_features
        }

        logger.info("Optimized feature subset datasets created successfully.")

        return subsets

    def _create_subset(
        self,
        bundle: DatasetBundle,
        features: list[str],
    ) -> DatasetBundle:
        """Create one immutable-style DatasetBundle feature subset.

        Args:
            bundle: Source dataset bundle.
            features: Ordered selected feature names.

        Returns:
            New dataset bundle containing only selected features.
        """

        metadata = bundle.metadata.copy()
        metadata["feature_count"] = len(features)
        metadata["selected_features"] = features.copy()

        return DatasetBundle(
            X_train=bundle.X_train.loc[:, features].copy(),
            y_train=bundle.y_train.copy(),
            X_valid=bundle.X_valid.loc[:, features].copy(),
            y_valid=bundle.y_valid.copy(),
            X_test=bundle.X_test.loc[:, features].copy(),
            y_test=bundle.y_test.copy(),
            feature_names=features.copy(),
            target_name=bundle.target_name,
            train_start=bundle.train_start,
            train_end=bundle.train_end,
            validation_start=bundle.validation_start,
            validation_end=bundle.validation_end,
            test_start=bundle.test_start,
            test_end=bundle.test_end,
            metadata=metadata,
        )
