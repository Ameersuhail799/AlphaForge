"""Model leaderboard generation for AlphaForge research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config.settings import (
    LEADERBOARD_REPORT_PATH,
    MODEL_COMPARISON_REPORT_PATH,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LeaderboardResult:
    """Contain the generated leaderboard and its output path."""

    leaderboard: pd.DataFrame
    report_path: Path


class LeaderboardManager:
    """Create a ranked model leaderboard from comparison metrics."""

    def generate(self) -> LeaderboardResult:
        """Generate and persist the model leaderboard.

        Returns:
            The sorted leaderboard and the saved report path.
        """

        logger.info("Generating model leaderboard...")

        leaderboard = self._load_comparison().copy()
        leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))

        LEADERBOARD_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(LEADERBOARD_REPORT_PATH, index=False)

        logger.info("Model leaderboard saved to %s.", LEADERBOARD_REPORT_PATH)

        return LeaderboardResult(
            leaderboard=leaderboard,
            report_path=LEADERBOARD_REPORT_PATH,
        )

    def _load_comparison(self) -> pd.DataFrame:
        """Load and rank the model comparison report.

        Returns:
            Comparison data sorted by ROC-AUC, F1, and Accuracy.

        Raises:
            FileNotFoundError: If the comparison report is missing.
        """

        if not MODEL_COMPARISON_REPORT_PATH.exists():
            raise FileNotFoundError(
                f"Comparison report not found: {MODEL_COMPARISON_REPORT_PATH}",
            )

        leaderboard = pd.read_csv(MODEL_COMPARISON_REPORT_PATH)
        leaderboard = leaderboard.sort_values(
            by=["ROC-AUC", "F1", "Accuracy"],
            ascending=[False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)

        return leaderboard