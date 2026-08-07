"""Champion model selection for AlphaForge research."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path

import pandas as pd

from config.settings import CHAMPION_REPORT_PATH
from src.research.leaderboard import LeaderboardManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChampionResult:
    """Contain the selected champion model and output path."""

    champion: dict[str, object]
    report_path: Path


class ChampionManager:
    """Select the best model from the comparison report."""

    def generate(self) -> ChampionResult:
        """Generate the champion JSON and leaderboard artifacts.

        Returns:
            Champion metadata and the saved JSON path.
        """

        logger.info("Selecting champion model...")

        leaderboard = LeaderboardManager().generate().leaderboard
        champion = self._select_champion(leaderboard)

        CHAMPION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHAMPION_REPORT_PATH.write_text(
            json.dumps(champion, indent=4),
            encoding="utf-8",
        )

        logger.info("Champion model saved to %s.", CHAMPION_REPORT_PATH)

        return ChampionResult(champion=champion, report_path=CHAMPION_REPORT_PATH)

    def _select_champion(self, leaderboard: pd.DataFrame) -> dict[str, object]:
        """Select the champion payload from the ranked leaderboard.

        Args:
            leaderboard: Comparison data sorted by ranking priority.

        Returns:
            Champion payload to persist.
        """

        champion_row = leaderboard.iloc[0]
        top_tied_rows = leaderboard[
            (leaderboard["ROC-AUC"] == champion_row["ROC-AUC"])
            & (leaderboard["F1"] == champion_row["F1"])
            & (leaderboard["Accuracy"] == champion_row["Accuracy"])
        ]
        existing_champion = self._load_existing_champion()

        if existing_champion is not None and self._is_same_tie(
            existing_champion,
            top_tied_rows,
        ):
            return existing_champion

        return self._build_champion_payload(champion_row, leaderboard)

    def _build_champion_payload(
        self,
        row: pd.Series,
        leaderboard: pd.DataFrame,
    ) -> dict[str, object]:
        """Convert the top leaderboard row into the champion payload.

        Args:
            row: Highest ranked comparison row.

        Returns:
            JSON-serializable champion metadata.
        """

        return {
            "model": row["Model"],
            "accuracy": float(row["Accuracy"]),
            "precision": float(row["Precision"]),
            "recall": float(row["Recall"]),
            "f1": float(row["F1"]),
            "roc_auc": float(row["ROC-AUC"]),
            "timestamp": self._deterministic_timestamp(leaderboard),
            "reason": self._reason(row, leaderboard),
        }

    def _load_existing_champion(self) -> dict[str, object] | None:
        """Load the existing champion payload if present.

        Returns:
            The persisted champion payload or ``None`` when unavailable.
        """

        if not CHAMPION_REPORT_PATH.exists():
            return None

        with CHAMPION_REPORT_PATH.open(encoding="utf-8") as file:
            return json.load(file)

    def _is_same_tie(
        self,
        existing_champion: dict[str, object],
        tied_rows: pd.DataFrame,
    ) -> bool:
        """Check whether the existing champion belongs to the exact tie set.

        Args:
            existing_champion: Previously persisted champion payload.
            tied_rows: Rows sharing the same top ranking metrics.

        Returns:
            ``True`` when the existing champion should be retained.
        """

        if not tied_rows.empty and existing_champion.get("model") in tied_rows[
            "Model"
        ].tolist():
            return (
                float(existing_champion["accuracy"]) == float(tied_rows.iloc[0]["Accuracy"])
                and float(existing_champion["precision"])
                == float(tied_rows.iloc[0]["Precision"])
                and float(existing_champion["recall"])
                == float(tied_rows.iloc[0]["Recall"])
                and float(existing_champion["f1"]) == float(tied_rows.iloc[0]["F1"])
                and float(existing_champion["roc_auc"])
                == float(tied_rows.iloc[0]["ROC-AUC"])
            )

        return False

    def _deterministic_timestamp(self, leaderboard: pd.DataFrame) -> str:
        """Derive a stable UTC timestamp from the ranked leaderboard.

        Args:
            leaderboard: Ranked model comparison table.

        Returns:
            Deterministic ISO 8601 timestamp string.
        """

        payload = leaderboard.to_csv(index=False).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        base_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
        offset_seconds = int.from_bytes(digest[:6], byteorder="big") % (50 * 365 * 24 * 60 * 60)
        offset_microseconds = int.from_bytes(digest[6:9], byteorder="big") % 1_000_000

        return (base_time + timedelta(seconds=offset_seconds, microseconds=offset_microseconds)).isoformat()

    def _reason(self, row: pd.Series, leaderboard: pd.DataFrame) -> str:
        """Explain why the model was selected.

        Args:
            row: Highest ranked comparison row.

        Returns:
            Human-readable selection reason.
        """

        top_roc_auc = float(leaderboard["ROC-AUC"].iloc[0])
        top_rows = leaderboard[leaderboard["ROC-AUC"] == top_roc_auc]

        if len(top_rows) == 1:
            return "Highest ROC-AUC"

        top_f1 = float(top_rows["F1"].max())
        f1_rows = top_rows[top_rows["F1"] == top_f1]

        if len(f1_rows) == 1:
            return "Highest ROC-AUC, then highest F1"

        top_accuracy = float(f1_rows["Accuracy"].max())
        if float(row["Accuracy"]) == top_accuracy:
            return "Highest ROC-AUC, then highest F1, then highest Accuracy"

        return "Highest ROC-AUC with tie resolved by F1 and Accuracy"