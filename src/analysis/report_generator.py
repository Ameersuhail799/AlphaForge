"""Baseline analysis report generation for AlphaForge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config.settings import ANALYSIS_REPORT_DIR
from src.analysis.class_distribution import (
    ClassDistributionAnalyzer,
    ClassDistributionResult,
)
from src.analysis.correlation_analysis import (
    CorrelationAnalyzer,
    CorrelationAnalysisResult,
)
from src.analysis.dataset_analysis import DatasetAnalysisResult, DatasetAnalyzer
from src.analysis.feature_analysis import FeatureAnalyzer
from src.dataset.bundle import DatasetBundle
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BaselineAnalysisResult:
    """Contain all baseline analysis outputs and artifact paths."""

    dataset_statistics: DatasetAnalysisResult
    class_distribution: ClassDistributionResult
    feature_statistics: pd.DataFrame
    correlation_analysis: CorrelationAnalysisResult
    report_path: Path


class ReportGenerator:
    """Orchestrate baseline analysis and generate a Markdown report."""

    def generate(
        self,
        bundle: DatasetBundle,
    ) -> BaselineAnalysisResult:
        """Generate baseline analysis artifacts and a Markdown summary.

        Args:
            bundle: Dataset bundle to analyze.

        Returns:
            Complete baseline analysis results and report location.
        """

        logger.info("Generating baseline analysis report...")

        dataset_statistics = DatasetAnalyzer().analyze(bundle)
        class_distribution = ClassDistributionAnalyzer().analyze(bundle)
        feature_statistics = FeatureAnalyzer().analyze(bundle)
        correlation_analysis = CorrelationAnalyzer().analyze(bundle)
        report_path = self._save_report(
            dataset_statistics,
            class_distribution,
            correlation_analysis,
        )

        result = BaselineAnalysisResult(
            dataset_statistics=dataset_statistics,
            class_distribution=class_distribution,
            feature_statistics=feature_statistics,
            correlation_analysis=correlation_analysis,
            report_path=report_path,
        )

        logger.info("Baseline analysis report generated successfully.")

        return result

    def _save_report(
        self,
        dataset_statistics: DatasetAnalysisResult,
        class_distribution: ClassDistributionResult,
        correlation_analysis: CorrelationAnalysisResult,
    ) -> Path:
        """Save a Markdown baseline analysis report.

        Args:
            dataset_statistics: Dataset-level baseline statistics.
            class_distribution: Target class distribution statistics.
            correlation_analysis: Correlation analysis outputs.

        Returns:
            Path to the saved Markdown report.
        """

        ANALYSIS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = ANALYSIS_REPORT_DIR / "baseline_report.md"
        strongest = correlation_analysis.target_correlation.abs().sort_values(
            ascending=False,
        ).head(5)
        weakest = correlation_analysis.target_correlation.abs().sort_values().head(5)
        recommendations = self._build_recommendations(
            class_distribution,
            strongest,
        )

        report = "\n".join(
            [
                "# AlphaForge Baseline Analysis",
                "",
                "## Dataset Statistics",
                "",
                f"- Total samples: {dataset_statistics.total_samples}",
                f"- Training samples: {dataset_statistics.training_samples}",
                f"- Validation samples: {dataset_statistics.validation_samples}",
                f"- Test samples: {dataset_statistics.test_samples}",
                f"- Feature count: {dataset_statistics.feature_count}",
                f"- Missing values: {dataset_statistics.missing_values}",
                f"- Duplicate rows: {dataset_statistics.duplicate_rows}",
                "",
                "## Target Balance",
                "",
                (
                    f"- Class 0: {class_distribution.class_zero_count} "
                    f"({class_distribution.class_zero_percentage:.2f}%)"
                ),
                (
                    f"- Class 1: {class_distribution.class_one_count} "
                    f"({class_distribution.class_one_percentage:.2f}%)"
                ),
                "",
                "## Correlation Summary",
                "",
                "- Pearson correlation matrix: `correlation_matrix.csv`",
                "- Target correlation ranking: `target_correlation.csv`",
                "- Correlation heatmap: `correlation_heatmap.png`",
                "",
                "## Strongest Features",
                "",
                *self._format_features(strongest),
                "",
                "## Weakest Features",
                "",
                *self._format_features(weakest),
                "",
                "## Recommendations",
                "",
                *recommendations,
                "",
            ]
        )

        report_path.write_text(report, encoding="utf-8")

        logger.info("Baseline report saved to %s.", report_path)

        return report_path

    def _format_features(self, features: pd.Series) -> list[str]:
        """Format feature-correlation values for Markdown output.

        Args:
            features: Feature correlation series.

        Returns:
            Markdown list items.
        """

        return [
            f"- `{feature}`: {correlation:.6f}"
            for feature, correlation in features.items()
        ]

    def _build_recommendations(
        self,
        class_distribution: ClassDistributionResult,
        strongest: pd.Series,
    ) -> list[str]:
        """Build data-driven recommendations for the baseline report.

        Args:
            class_distribution: Target class balance information.
            strongest: Highest absolute target correlations.

        Returns:
            Markdown recommendation list items.
        """

        balance_gap = abs(
            class_distribution.class_zero_percentage
            - class_distribution.class_one_percentage
        )
        recommendations = [
            (
                "- Validate the highest-ranked features with walk-forward "
                "modelling before relying on their apparent signal."
            ),
            (
                "- Review highly correlated feature pairs before using models "
                "sensitive to multicollinearity."
            ),
        ]

        if balance_gap > 10:
            recommendations.append(
                "- Consider class-aware evaluation because target classes are "
                "materially imbalanced."
            )
        else:
            recommendations.append(
                "- Target classes are sufficiently balanced for standard "
                "classification metrics."
            )

        if strongest.iloc[0] < 0.05:
            recommendations.append(
                "- Low individual target correlations suggest evaluating "
                "non-linear models and feature interactions."
            )

        return recommendations
