"""Prediction API for AlphaForge machine learning models."""

from __future__ import annotations

from time import perf_counter

import pandas as pd

from src.dataset.bundle import DatasetBundle
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Generate test-set predictions from a trained AlphaForge model."""

    def __init__(self) -> None:
        """Initialize prediction timing state."""

        self.prediction_time_seconds = 0.0

    def predict(
        self,
        model: BaseModel,
        bundle: DatasetBundle,
    ) -> pd.Series:
        """Return predicted classes for the dataset bundle test split.

        Args:
            model: Trained AlphaForge model.
            bundle: Dataset bundle containing test features.

        Returns:
            Indexed predicted classes.
        """

        logger.info("Generating model predictions...")

        start_time = perf_counter()
        predictions = pd.Series(
            model.predict(bundle),
            index=bundle.X_test.index,
            name="prediction",
        )
        self.prediction_time_seconds = perf_counter() - start_time

        logger.info("Model predictions generated successfully.")

        return predictions

    def predict_probabilities(
        self,
        model: BaseModel,
        bundle: DatasetBundle,
    ) -> pd.Series:
        """Return positive-class probabilities for the test split.

        Args:
            model: Trained AlphaForge model.
            bundle: Dataset bundle containing test features.

        Returns:
            Indexed positive-class probabilities.
        """

        logger.info("Generating model probabilities...")

        probabilities = pd.Series(
            model.predict_proba(bundle),
            index=bundle.X_test.index,
            name="probability",
        )

        logger.info("Model probabilities generated successfully.")

        return probabilities
