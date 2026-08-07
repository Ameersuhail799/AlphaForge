"""Feature distribution analysis for AlphaForge."""

from __future__ import annotations

import pandas as pd

from src.analysis.dataset_analysis import combine_bundle
from src.dataset.bundle import DatasetBundle
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureAnalyzer:
    """Calculate descriptive statistics for every dataset feature."""

    def analyze(
        self,
        bundle: DatasetBundle,
    ) -> pd.DataFrame:
        """Calculate descriptive feature statistics.

        Args:
            bundle: Dataset bundle containing feature partitions.

        Returns:
            DataFrame with one row per feature and requested statistics.
        """

        logger.info("Analyzing feature distributions...")

        features, _ = combine_bundle(bundle)
        statistics = pd.DataFrame(
            {
                "mean": features.mean(),
                "median": features.median(),
                "std": features.std(),
                "min": features.min(),
                "max": features.max(),
                "variance": features.var(),
                "skewness": features.skew(),
                "kurtosis": features.kurt(),
            }
        )

        logger.info("Feature distribution analysis completed successfully.")

        return statistics
