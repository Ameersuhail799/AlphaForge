"""Training, persistence, and experiment reporting for AlphaForge models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib

from config.settings import (
    EXPERIMENT_REPORT_DIR,
    FEATURE_IMPORTANCE_REPORT_DIR,
    TRAINED_MODEL_DIR,
)
from src.dataset.bundle import DatasetBundle
from src.models.base_model import BaseModel
from src.models.metrics import ModelMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Train models and persist their artifacts and experiment records."""

    def __init__(self) -> None:
        """Initialize trainer state."""

        self.model: BaseModel | None = None

    def train(
        self,
        model: BaseModel,
        bundle: DatasetBundle,
    ) -> BaseModel:
        """Train a model using the training split from a dataset bundle.

        Args:
            model: AlphaForge model to train.
            bundle: Dataset bundle containing training features and targets.

        Returns:
            The trained model.
        """

        logger.info("Starting training for %s...", model.model_name)

        model.fit(bundle)
        self.model = model

        logger.info("Training completed for %s.", model.model_name)

        return model

    def save_model(
        self,
        model: BaseModel | None = None,
        experiment_id: str | None = None,
    ) -> Path:
        """Save a trained model as a joblib artifact.

        Args:
            model: Trained model to save. Uses the stored model when omitted.
            experiment_id: Identifier included in the artifact file name.

        Returns:
            Path to the saved joblib model.

        Raises:
            RuntimeError: If no trained model is available.
        """

        model_to_save = model or self.model

        if model_to_save is None:
            raise RuntimeError("No trained model is available to save.")

        artifact_id = experiment_id or self._create_experiment_id(
            model_to_save.model_name
        )
        TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = TRAINED_MODEL_DIR / f"{artifact_id}.joblib"

        joblib.dump(model_to_save, model_path)

        logger.info("Model saved to %s.", model_path)

        return model_path

    def load_model(self, model_path: Path) -> BaseModel:
        """Load a previously saved joblib model.

        Args:
            model_path: Path to the joblib model artifact.

        Returns:
            Loaded AlphaForge model.
        """

        logger.info("Loading model from %s...", model_path)
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")

        return model

    def export_feature_importance(
        self,
        model: BaseModel | None = None,
        experiment_id: str | None = None,
    ) -> Path:
        """Export sorted model feature importance as CSV.

        Args:
            model: Trained model to inspect. Uses the stored model when omitted.
            experiment_id: Identifier included in the report file name.

        Returns:
            Path to the feature importance report.

        Raises:
            RuntimeError: If no trained model is available.
        """

        model_to_export = model or self.model

        if model_to_export is None:
            raise RuntimeError("No trained model is available to inspect.")

        report_id = experiment_id or self._create_experiment_id(
            model_to_export.model_name
        )
        FEATURE_IMPORTANCE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = FEATURE_IMPORTANCE_REPORT_DIR / (
            f"{report_id}_feature_importance.csv"
        )

        model_to_export.feature_importance().to_csv(report_path, index=False)

        logger.info("Feature importance saved to %s.", report_path)

        return report_path

    def save_experiment(
        self,
        model: BaseModel,
        bundle: DatasetBundle,
        metrics: ModelMetrics,
        experiment_id: str | None = None,
    ) -> Path:
        """Save experiment metadata and evaluation results as JSON.

        Args:
            model: Trained AlphaForge model.
            bundle: Dataset bundle used for the experiment.
            metrics: Evaluation metrics from the test partition.
            experiment_id: Optional identifier for the experiment.

        Returns:
            Path to the experiment JSON file.
        """

        report_id = experiment_id or self._create_experiment_id(
            model.model_name
        )
        EXPERIMENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = EXPERIMENT_REPORT_DIR / f"{report_id}.json"

        experiment = {
            "experiment_id": report_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model.model_name,
            "parameters": model.get_parameters(),
            "feature_count": len(bundle.feature_names),
            "training_samples": len(bundle.X_train),
            "validation_samples": len(bundle.X_valid),
            "test_samples": len(bundle.X_test),
            **metrics.to_dict(),
        }

        with report_path.open("w", encoding="utf-8") as file:
            json.dump(experiment, file, indent=2)

        logger.info("Experiment metadata saved to %s.", report_path)

        return report_path

    def _create_experiment_id(self, model_name: str) -> str:
        """Create a unique, traceable experiment identifier.

        Args:
            model_name: Name of the trained model.

        Returns:
            Unique experiment identifier.
        """

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        return f"{model_name}_{timestamp}_{uuid4().hex[:8]}"
