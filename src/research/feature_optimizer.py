"""Feature importance aggregation for AlphaForge research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.settings import FEATURE_IMPORTANCE_REPORT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureOptimizationResult:
    """Contain aggregated feature rankings and top feature subsets."""

    rankings: pd.DataFrame
    top_features: dict[int, list[str]]


class FeatureOptimizer:
    """Aggregate model feature importance files into shared feature rankings."""

    TOP_COUNTS = [5, 10, 15, 20]

    def optimize(self) -> FeatureOptimizationResult:
        """Read model importance reports and aggregate their feature ranks.

        Returns:
            Aggregated ranking table and top feature lists.

        Raises:
            FileNotFoundError: If no feature importance reports are available.
            ValueError: If an importance report has invalid columns.
        """

        logger.info("Aggregating model feature importance rankings...")

        ranking_rows = self._load_rankings()
        rankings = (
            ranking_rows.groupby("feature", as_index=False)
            .agg(
                average_rank=("rank", "mean"),
                model_count=("model", "nunique"),
                average_importance=("importance", "mean"),
            )
            .sort_values(
                ["average_rank", "average_importance"],
                ascending=[True, False],
            )
            .reset_index(drop=True)
        )
        top_features = {
            count: rankings.head(count)["feature"].to_list()
            for count in self.TOP_COUNTS
        }

        logger.info("Feature importance aggregation completed successfully.")

        return FeatureOptimizationResult(
            rankings=rankings,
            top_features=top_features,
        )

    def _load_rankings(self) -> pd.DataFrame:
        """Load and rank every model feature importance report.

        Returns:
            Normalized per-model feature ranking records.

        Raises:
            FileNotFoundError: If no importance reports are available.
            ValueError: If report columns are invalid.
        """

        report_paths = sorted(FEATURE_IMPORTANCE_REPORT_DIR.glob("*.csv"))

        if not report_paths:
            raise FileNotFoundError("No feature importance reports were found.")

        ranking_frames: list[pd.DataFrame] = []

        for report_path in report_paths:
            importance = pd.read_csv(report_path)

            if not {"feature", "importance"}.issubset(importance.columns):
                raise ValueError(
                    f"Invalid feature importance report: {report_path}"
                )

            ranked = importance[["feature", "importance"]].copy()
            ranked["model"] = report_path.stem.replace(
                "_feature_importance",
                "",
            )
            ranked["rank"] = ranked["importance"].rank(
                ascending=False,
                method="min",
            )
            ranking_frames.append(ranked)

        return pd.concat(ranking_frames, ignore_index=True)
