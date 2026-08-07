"""Correlation analysis and visualization for AlphaForge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.settings import ANALYSIS_REPORT_DIR
from src.analysis.dataset_analysis import combine_bundle
from src.dataset.bundle import DatasetBundle
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CorrelationAnalysisResult:
    """Contain correlation outputs, rankings, and artifact paths."""

    correlation_matrix: pd.DataFrame
    target_correlation: pd.Series
    top_positive_features: pd.Series
    top_negative_features: pd.Series
    matrix_path: Path
    target_path: Path
    heatmap_path: Path


class CorrelationAnalyzer:
    """Calculate Pearson correlations and export correlation artifacts."""

    def analyze(
        self,
        bundle: DatasetBundle,
    ) -> CorrelationAnalysisResult:
        """Calculate feature correlations and save CSV and heatmap outputs.

        Args:
            bundle: Dataset bundle containing feature and target partitions.

        Returns:
            Correlation matrices, target rankings, and artifact paths.
        """

        logger.info("Analyzing feature correlations...")

        features, target = combine_bundle(bundle)
        correlation_matrix = features.corr(method="pearson")
        target_correlation = features.corrwith(target).sort_values(
            ascending=False,
        )
        top_positive = target_correlation[target_correlation > 0].head(5)
        top_negative = target_correlation[target_correlation < 0].sort_values().head(5)

        matrix_path, target_path = self._save_csv(
            correlation_matrix,
            target_correlation,
        )
        heatmap_path = self._save_heatmap(correlation_matrix)

        result = CorrelationAnalysisResult(
            correlation_matrix=correlation_matrix,
            target_correlation=target_correlation,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            matrix_path=matrix_path,
            target_path=target_path,
            heatmap_path=heatmap_path,
        )

        logger.info("Correlation analysis completed successfully.")

        return result

    def _save_csv(
        self,
        correlation_matrix: pd.DataFrame,
        target_correlation: pd.Series,
    ) -> tuple[Path, Path]:
        """Save correlation matrix and target correlation CSV files.

        Args:
            correlation_matrix: Pearson feature correlation matrix.
            target_correlation: Correlation of each feature with the target.

        Returns:
            Paths to the matrix and target-correlation CSV files.
        """

        ANALYSIS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        matrix_path = ANALYSIS_REPORT_DIR / "correlation_matrix.csv"
        target_path = ANALYSIS_REPORT_DIR / "target_correlation.csv"

        correlation_matrix.to_csv(matrix_path)
        target_correlation.rename("correlation").to_csv(target_path)

        logger.info("Correlation CSV files saved to %s.", ANALYSIS_REPORT_DIR)

        return matrix_path, target_path

    def _save_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
    ) -> Path:
        """Save a Pearson correlation heatmap PNG.

        Args:
            correlation_matrix: Pearson feature correlation matrix.

        Returns:
            Path to the saved heatmap PNG.
        """

        ANALYSIS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        heatmap_path = ANALYSIS_REPORT_DIR / "correlation_heatmap.png"

        figure, axis = plt.subplots(figsize=(16, 14))
        image = axis.imshow(
            correlation_matrix,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        axis.set_title("Pearson Feature Correlation Matrix")
        axis.set_xticks(range(len(correlation_matrix.columns)))
        axis.set_xticklabels(
            correlation_matrix.columns,
            rotation=90,
            fontsize=7,
        )
        axis.set_yticks(range(len(correlation_matrix.index)))
        axis.set_yticklabels(correlation_matrix.index, fontsize=7)
        figure.colorbar(image, ax=axis, label="Correlation")

        figure.tight_layout()
        figure.savefig(heatmap_path, dpi=150)
        plt.close(figure)

        logger.info("Correlation heatmap saved to %s.", heatmap_path)

        return heatmap_path
