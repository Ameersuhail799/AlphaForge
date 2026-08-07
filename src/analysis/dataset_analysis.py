"""Dataset-level baseline analysis for AlphaForge."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.dataset.bundle import DatasetBundle
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetAnalysisResult:
    """Contain dataset-level baseline statistics."""

    total_samples: int
    training_samples: int
    validation_samples: int
    test_samples: int
    feature_count: int
    missing_values: int
    duplicate_rows: int
    target_balance: dict[int, int]


def combine_bundle(
    bundle: DatasetBundle,
) -> tuple[pd.DataFrame, pd.Series]:
    """Combine chronological DatasetBundle partitions for analysis.

    Args:
        bundle: Dataset bundle to analyze.

    Returns:
        Combined feature matrix and aligned target series.
    """

    features = pd.concat(
        [bundle.X_train, bundle.X_valid, bundle.X_test],
    )
    target = pd.concat(
        [bundle.y_train, bundle.y_valid, bundle.y_test],
    )

    return features, target


class DatasetAnalyzer:
    """Generate baseline sample and data-quality statistics."""

    def analyze(
        self,
        bundle: DatasetBundle,
    ) -> DatasetAnalysisResult:
        """Analyze DatasetBundle sample counts and data quality.

        Args:
            bundle: Dataset bundle to inspect.

        Returns:
            Dataset-level baseline statistics.
        """

        logger.info("Analyzing dataset statistics...")

        features, target = combine_bundle(bundle)
        missing_values = int(features.isna().sum().sum() + target.isna().sum())

        result = DatasetAnalysisResult(
            total_samples=len(features),
            training_samples=len(bundle.X_train),
            validation_samples=len(bundle.X_valid),
            test_samples=len(bundle.X_test),
            feature_count=len(bundle.feature_names),
            missing_values=missing_values,
            duplicate_rows=int(features.duplicated().sum()),
            target_balance={
                int(label): int(count)
                for label, count in target.value_counts().sort_index().items()
            },
        )

        logger.info("Dataset analysis completed successfully.")

        return result
