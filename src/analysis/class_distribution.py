"""Class distribution analysis and visualization for AlphaForge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.settings import ANALYSIS_REPORT_DIR
from src.dataset.bundle import DatasetBundle
from src.analysis.dataset_analysis import combine_bundle
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClassDistributionResult:
    """Contain class counts, percentages, and plot location."""

    class_zero_count: int
    class_one_count: int
    class_zero_percentage: float
    class_one_percentage: float
    plot_path: Path


class ClassDistributionAnalyzer:
    """Calculate and visualize binary target distribution."""

    def analyze(
        self,
        bundle: DatasetBundle,
    ) -> ClassDistributionResult:
        """Analyze target balance and save a class distribution chart.

        Args:
            bundle: Dataset bundle containing target partitions.

        Returns:
            Target distribution statistics and chart path.
        """

        logger.info("Analyzing target class distribution...")

        _, target = combine_bundle(bundle)
        counts = target.value_counts().reindex([0, 1], fill_value=0)
        total = len(target)
        plot_path = self._save_plot(counts)

        result = ClassDistributionResult(
            class_zero_count=int(counts[0]),
            class_one_count=int(counts[1]),
            class_zero_percentage=float(counts[0] / total * 100),
            class_one_percentage=float(counts[1] / total * 100),
            plot_path=plot_path,
        )

        logger.info("Class distribution analysis completed successfully.")

        return result

    def _save_plot(self, counts: pd.Series) -> Path:
        """Save a target class distribution bar chart.

        Args:
            counts: Counts for class zero and class one.

        Returns:
            Path to the saved PNG chart.
        """

        ANALYSIS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = ANALYSIS_REPORT_DIR / "class_distribution.png"

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(
            ["Class 0", "Class 1"],
            counts.to_list(),
            color=["#4c72b0", "#dd8452"],
        )
        axis.set_title("Target Class Distribution")
        axis.set_xlabel("Target Class")
        axis.set_ylabel("Samples")

        figure.tight_layout()
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)

        logger.info("Class distribution plot saved to %s.", plot_path)

        return plot_path
