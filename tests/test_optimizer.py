"""Integration test for hyperparameter optimization."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import (
    CHAMPION_REPORT_PATH,
    EXPERIMENT_HISTORY_REPORT_PATH,
    LEADERBOARD_REPORT_PATH,
    MODEL_COMPARISON_REPORT_PATH,
    REPORT_DIR,
)

from src.dataset.builder import DatasetBuilder
from src.features.feature_pipeline import FeaturePipeline
from src.research.optimizer import HyperparameterOptimizer


def _build_test_frame(rows: int = 180) -> pd.DataFrame:
    """Build deterministic OHLCV data for optimization testing.

    Args:
        rows: Number of time steps to generate.

    Returns:
        Deterministic market-style data frame.
    """

    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    closes: list[float] = []
    value = 100.0

    for index in range(rows):
        if (index // 6) % 2 == 0:
            value += 1.5
        else:
            value -= 1.5
        closes.append(value)

    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": [close - 0.5 for close in closes],
            "High": [close + 1.0 for close in closes],
            "Low": [close - 1.0 for close in closes],
            "Close": closes,
            "Volume": [1_000_000 + (index % 10) * 25_000 for index in range(rows)],
        }
    )

    return frame.set_index("Date")


def main() -> None:
    """Run the optimizer and assert the generated artifacts."""

    original_champion = CHAMPION_REPORT_PATH.read_text(encoding="utf-8") if CHAMPION_REPORT_PATH.exists() else None
    original_comparison = MODEL_COMPARISON_REPORT_PATH.read_text(encoding="utf-8") if MODEL_COMPARISON_REPORT_PATH.exists() else None
    original_history = EXPERIMENT_HISTORY_REPORT_PATH.read_text(encoding="utf-8") if EXPERIMENT_HISTORY_REPORT_PATH.exists() else None
    original_leaderboard = LEADERBOARD_REPORT_PATH.read_text(encoding="utf-8") if LEADERBOARD_REPORT_PATH.exists() else None

    low_champion = {
        "model": "baseline",
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": "Test baseline",
    }

    CHAMPION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAMPION_REPORT_PATH.write_text(json.dumps(low_champion, indent=4), encoding="utf-8")

    try:
        df = _build_test_frame()
        engineered_df = FeaturePipeline().generate(df)
        bundle = DatasetBuilder(scale=True).build(
            engineered_df,
            symbol="RELIANCE.NS",
        )

        optimizer = HyperparameterOptimizer(n_iter=2, random_state=42)
        result = optimizer.optimize("random_forest", bundle)

        report_path = REPORT_DIR / "optimization" / "random_forest_optimization.json"

        assert report_path.exists()
        assert result.best_parameters
        assert result.best_score is not None
        assert not result.history.empty

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["best_parameters"] == result.best_parameters
        assert report.get("optimization_metric") == result.best_score
        assert len(report["evaluated_combinations"]) == len(result.history)

        # Report must contain both validation and final test scores
        assert "validation_score" in report
        assert "final_test_score" in report

        # Champion payload should reflect VALIDATION metrics used for selection
        champion_payload = json.loads(CHAMPION_REPORT_PATH.read_text(encoding="utf-8"))
        val = report["validation_score"]
        assert float(champion_payload["accuracy"]) == float(val["accuracy"])
        assert float(champion_payload["f1"]) == float(val["f1"])
        # optimization report final_test_score must match the model_comparison (TEST) row
        test = report["final_test_score"]

        with MODEL_COMPARISON_REPORT_PATH.open(encoding="utf-8", newline="") as file:
            comparison_rows = list(csv.DictReader(file))

        # Find the comparison row for the optimized model and compare TEST metrics
        matching = [r for r in comparison_rows if r["Model"] == champion_payload["model"]]
        assert matching, "Optimized model missing from comparison report"
        comp = matching[0]
        assert abs(float(comp["ROC-AUC"]) - float(test["roc_auc"])) < 1e-8

        assert result.champion_updated is True
        assert EXPERIMENT_HISTORY_REPORT_PATH.exists()
        assert LEADERBOARD_REPORT_PATH.exists()

        print("OPTIMIZER SUCCESS")
        print(result.best_parameters)
    finally:
        if original_champion is None:
            CHAMPION_REPORT_PATH.unlink(missing_ok=True)
        else:
            CHAMPION_REPORT_PATH.write_text(original_champion, encoding="utf-8")

        if original_comparison is None:
            MODEL_COMPARISON_REPORT_PATH.unlink(missing_ok=True)
        else:
            MODEL_COMPARISON_REPORT_PATH.write_text(original_comparison, encoding="utf-8")

        if original_history is None:
            EXPERIMENT_HISTORY_REPORT_PATH.unlink(missing_ok=True)
        else:
            EXPERIMENT_HISTORY_REPORT_PATH.write_text(original_history, encoding="utf-8")

        if original_leaderboard is None:
            LEADERBOARD_REPORT_PATH.unlink(missing_ok=True)
        else:
            LEADERBOARD_REPORT_PATH.write_text(original_leaderboard, encoding="utf-8")


if __name__ == "__main__":
    main()