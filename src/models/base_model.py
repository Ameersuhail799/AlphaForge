"""Base interfaces for AlphaForge machine learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.dataset.bundle import DatasetBundle


class BaseModel(ABC):
    """Define the dataset-bundle-only interface for AlphaForge models."""

    model_name: str

    @abstractmethod
    def fit(self, bundle: DatasetBundle) -> None:
        """Fit the model using the training partition in a dataset bundle.

        Args:
            bundle: Dataset bundle containing training features and targets.
        """

    @abstractmethod
    def predict(self, bundle: DatasetBundle) -> np.ndarray:
        """Predict classes for the test partition in a dataset bundle.

        Args:
            bundle: Dataset bundle containing test features.

        Returns:
            Predicted target classes.
        """

    @abstractmethod
    def predict_proba(self, bundle: DatasetBundle) -> np.ndarray:
        """Predict positive-class probabilities for the test partition.

        Args:
            bundle: Dataset bundle containing test features.

        Returns:
            Positive-class probabilities.
        """

    @abstractmethod
    def feature_importance(self) -> pd.DataFrame:
        """Return model feature importance ordered from highest to lowest.

        Returns:
            Feature importance data.
        """

    @abstractmethod
    def get_parameters(self) -> dict[str, object]:
        """Return the configured model parameters.

        Returns:
            Model parameter mapping.
        """
