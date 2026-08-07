"""Integration test for champion model selection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.settings import (
    CHAMPION_REPORT_PATH,
    LEADERBOARD_REPORT_PATH,
    MODEL_COMPARISON_REPORT_PATH,
)
from src.research.champion import ChampionManager


def main() -> None:
    """Generate champion artifacts and assert the selected model."""

    result = ChampionManager().generate()
    first_champion_text = CHAMPION_REPORT_PATH.read_text(encoding="utf-8")
    first_leaderboard_text = LEADERBOARD_REPORT_PATH.read_text(encoding="utf-8")

    second_result = ChampionManager().generate()
    second_champion_text = CHAMPION_REPORT_PATH.read_text(encoding="utf-8")
    second_leaderboard_text = LEADERBOARD_REPORT_PATH.read_text(encoding="utf-8")

    assert CHAMPION_REPORT_PATH.exists()
    assert LEADERBOARD_REPORT_PATH.exists()
    assert first_champion_text == second_champion_text
    assert first_leaderboard_text == second_leaderboard_text

    comparison = pd.read_csv(MODEL_COMPARISON_REPORT_PATH)
    expected = comparison.sort_values(
        by=["ROC-AUC", "F1", "Accuracy"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]

    champion_payload = json.loads(second_champion_text)
    leaderboard = pd.read_csv(LEADERBOARD_REPORT_PATH)

    assert champion_payload["model"] == expected["Model"]
    assert result.champion["model"] == expected["Model"]
    assert second_result.champion["model"] == expected["Model"]
    assert leaderboard.iloc[0]["Model"] == expected["Model"]
    assert list(leaderboard["Rank"]) == list(range(1, len(leaderboard) + 1))
    assert leaderboard["ROC-AUC"].is_monotonic_decreasing

    print("CHAMPION SYSTEM SUCCESS")
    print(champion_payload)
    print(leaderboard.to_dict(orient="records"))


if __name__ == "__main__":
    main()