"""Integration test to run a real-market comparison between models.

This test executes a controlled experiment using the stored
`RELIANCE_NS` dataset and the project's existing training
and evaluation pipeline. It does not modify the production
champion artifact.
"""

from __future__ import annotations

import json
from time import perf_counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.storage import StorageEngine
from src.features.feature_pipeline import FeaturePipeline
from src.dataset.builder import DatasetBuilder
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
from src.models.evaluator import Evaluator
from src.utils.logger import get_logger
from config.settings import REPORT_DIR

logger = get_logger(__name__)


def _evaluate_on_matrix(evaluator: Evaluator, y_true, preds, probs):
    return evaluator.evaluate(type("B", (), {"y_test": y_true}), preds, probs)


def main() -> None:
    storage = StorageEngine()

    # Load stored dataset
    dataset_name = "RELIANCE_NS"
    if not storage.dataset_exists(dataset_name):
        raise FileNotFoundError(
            f"Stored dataset '{dataset_name}' not found in {storage.base_dir}."
        )

    df = storage.load_dataset(dataset_name)

    # Feature engineering and dataset construction
    engineered = FeaturePipeline().generate(df)
    bundle = DatasetBuilder(scale=True).build(engineered, symbol="RELIANCE.NS")

    registry = ModelRegistry()
    evaluator = Evaluator()

    models = ["logistic_regression", "random_forest", "xgboost"]
    results: dict[str, dict[str, Any]] = {}

    # Train on TRAIN and evaluate on VALIDATION for model selection
    for name in models:
        model = registry.create(name)
        trainer = Trainer()

        # Train on TRAIN
        trainer.train(model, bundle)
        training_time = trainer.training_time_seconds

        # Validation predictions (directly against underlying estimator)
        estimator = getattr(model, "model")
        start = perf_counter()
        val_preds = estimator.predict(bundle.X_valid)
        val_probs = estimator.predict_proba(bundle.X_valid)[:, 1]
        prediction_time = perf_counter() - start

        val_metrics = _evaluate_on_matrix(evaluator, bundle.y_valid, val_preds, val_probs)

        results[name] = {
            "model": model,
            "trainer": trainer,
            "validation_metrics": val_metrics.to_dict(),
            "validation_rank": (
                float(val_metrics.roc_auc) if val_metrics.roc_auc is not None else float("-inf"),
                float(val_metrics.f1),
                float(val_metrics.accuracy),
            ),
            "validation_training_time": training_time,
            "validation_prediction_time": prediction_time,
        }

    # Select champion based on VALIDATION metrics only (ROC-AUC, F1, Accuracy)
    champion_name = max(results.keys(), key=lambda n: results[n]["validation_rank"])

    # Prepare combined TRAIN+VALID bundle for retraining
    X_combined = pd.concat([bundle.X_train, bundle.X_valid])
    y_combined = pd.concat([bundle.y_train, bundle.y_valid])

    # Build a DatasetBundle-like minimal object for trainer.fit usage
    from src.dataset.bundle import DatasetBundle

    combined = DatasetBundle(
        X_train=X_combined,
        y_train=y_combined,
        X_valid=bundle.X_valid,
        y_valid=bundle.y_valid,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        feature_names=bundle.feature_names,
        target_name=bundle.target_name,
        train_start=bundle.train_start,
        train_end=bundle.validation_end,
        validation_start=bundle.validation_start,
        validation_end=bundle.validation_end,
        test_start=bundle.test_start,
        test_end=bundle.test_end,
        metadata=bundle.metadata,
    )

    # Ensure no test metrics exist before champion selection (safety check)
    for n in models:
        assert "test_metrics" not in results[n]

    # Retrain ALL models on TRAIN+VALID and evaluate once on TEST
    for name in models:
        # Retrain a fresh model instance on the combined data to keep retraining independent
        m = registry.create(name)
        retrain_trainer = Trainer()
        retrain_trainer.train(m, combined)

        # Test evaluation
        estimator = getattr(m, "model")
        start = perf_counter()
        t_preds = estimator.predict(bundle.X_test)
        t_probs = estimator.predict_proba(bundle.X_test)[:, 1]
        t_prediction_time = perf_counter() - start

        t_metrics = _evaluate_on_matrix(evaluator, bundle.y_test, t_preds, t_probs)

        # Record test metrics and retrain metadata
        results[name]["test_metrics"] = t_metrics.to_dict()
        results[name]["test_training_time"] = retrain_trainer.training_time_seconds
        results[name]["test_prediction_time"] = t_prediction_time
        results[name]["test_training_samples"] = len(X_combined)

    # For convenience, expose champion variable (selected earlier using validation only)
    champ_name = champion_name

    # Save feature importance for RF and XGBoost
    FEATURE_DIR = REPORT_DIR / "feature_importance"
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # Save feature importance for RF and XGBoost (trained on TRAIN+VALID)
    FEATURE_DIR = REPORT_DIR / "feature_importance"
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    for name in ["random_forest", "xgboost"]:
        m = registry.create(name)
        trainer = Trainer()
        trainer.train(m, combined)
        trainer.export_feature_importance(m, experiment_id=f"{name}_real_market")

    # Build CSV report
    rows = []
    for name in models:
        val = results[name]["validation_metrics"]
        final = results[name].get("test_metrics", {})
        training_time = results[name].get("test_training_time", results[name].get("validation_training_time"))
        prediction_time = results[name].get("test_prediction_time", results[name].get("validation_prediction_time"))
        training_samples = results[name].get("test_training_samples", len(bundle.X_train))

        rows.append(
            {
                "Model": name,
                "Validation Accuracy": val["accuracy"],
                "Validation Precision": val["precision"],
                "Validation Recall": val["recall"],
                "Validation F1": val["f1"],
                "Validation ROC-AUC": val.get("roc_auc"),
                "Test Accuracy": final.get("accuracy"),
                "Test Precision": final.get("precision"),
                "Test Recall": final.get("recall"),
                "Test F1": final.get("f1"),
                "Test ROC-AUC": final.get("roc_auc"),
                "Training Time": training_time,
                "Prediction Time": prediction_time,
                "Feature Count": len(bundle.feature_names),
                "Training Samples": training_samples,
                "Validation Samples": len(bundle.X_valid),
                "Test Samples": len(bundle.X_test),
            }
        )

    # Persist reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "real_market_model_comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    json_path = REPORT_DIR / "experiments" / "real_market_comparison.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"models": rows, "champion": champion_name}, f, indent=2)

    # Basic assertions for the test harness
    assert csv_path.exists()
    assert json_path.exists()
    for name in ["random_forest", "xgboost"]:
        fi = REPORT_DIR / "feature_importance" / f"{name}_real_market_feature_importance.csv"
        assert fi.exists()

    print("REAL MARKET EXPERIMENT SUCCESS")


if __name__ == "__main__":
    main()
