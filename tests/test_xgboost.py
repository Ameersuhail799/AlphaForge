"""Integration test for XGBoost model integration."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from config.settings import CHAMPION_REPORT_PATH, EXPERIMENT_HISTORY_REPORT_PATH, REPORT_DIR, MODEL_COMPARISON_REPORT_PATH

from src.features.feature_pipeline import FeaturePipeline
from src.dataset.builder import DatasetBuilder
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
from src.models.predictor import Predictor
from src.models.evaluator import Evaluator


def _build_test_frame(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    closes = [100.0 + (i % 10) * 0.5 for i in range(rows)]

    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": [c - 0.2 for c in closes],
            "High": [c + 0.5 for c in closes],
            "Low": [c - 0.5 for c in closes],
            "Close": closes,
            "Volume": [100000 + (i % 5) * 1000 for i in range(rows)],
        }
    )

    return frame.set_index("Date")


def main() -> None:
    original_champion = CHAMPION_REPORT_PATH.read_text(encoding="utf-8") if CHAMPION_REPORT_PATH.exists() else None

    try:
        df = _build_test_frame()
        engineered = FeaturePipeline().generate(df)
        bundle = DatasetBuilder(scale=True).build(engineered, symbol="TEST.SYMBOL")

        registry = ModelRegistry()
        assert "xgboost" in registry.list_models()

        model = registry.create("xgboost")

        trainer = Trainer()
        trainer.train(model, bundle)

        predictor = Predictor()
        preds = predictor.predict(model, bundle)
        probs = predictor.predict_probabilities(model, bundle)

        metrics = Evaluator().evaluate(bundle, preds.to_numpy(), probs.to_numpy())

        # Export feature importance
        fi_path = trainer.export_feature_importance(model, experiment_id="xgboost_integration_test")
        assert fi_path.exists()

        # Save experiment using existing workflow (uses TEST metrics)
        trainer.save_experiment(model, bundle, metrics, experiment_id="xgboost_integration_test")

        # Ensure model comparison updated
        assert MODEL_COMPARISON_REPORT_PATH.exists()

        print("XGBOOST SUCCESS")

    finally:
        if original_champion is None:
            CHAMPION_REPORT_PATH.unlink(missing_ok=True)
        else:
            CHAMPION_REPORT_PATH.write_text(original_champion, encoding="utf-8")


if __name__ == "__main__":
    main()
