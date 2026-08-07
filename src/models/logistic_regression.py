"""Logistic Regression model implementation for AlphaForge."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.dataset.bundle import DatasetBundle
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LogisticRegressionModel(BaseModel):
    """Train and serve a Logistic Regression classifier from a dataset bundle."""

    model_name = "logistic_regression"

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
        **parameters: object,
    ) -> None:
        """Initialize the sklearn Logistic Regression estimator.

        Args:
            random_state: Seed used by the underlying estimator.
            max_iter: Maximum optimization iterations.
            **parameters: Additional sklearn LogisticRegression parameters.
        """

        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            **parameters,
        )
        self._feature_names: list[str] = []
        self._is_trained = False

    def fit(self, bundle: DatasetBundle) -> None:
        """Fit Logistic Regression on a dataset bundle's training partition.

        Args:
            bundle: Dataset bundle containing training features and targets.
        """

        logger.info("Training Logistic Regression model...")

        self.model.fit(bundle.X_train, bundle.y_train)
        self._feature_names = bundle.feature_names.copy()
        self._is_trained = True

        logger.info("Logistic Regression model trained successfully.")

    def predict(self, bundle: DatasetBundle) -> np.ndarray:
        """Predict target classes for a dataset bundle's test partition.

        Args:
            bundle: Dataset bundle containing test features.

        Returns:
            Predicted target classes.

        Raises:
            RuntimeError: If the model has not been trained.
        """

        self._validate_trained()

        return self.model.predict(bundle.X_test)

    def predict_proba(self, bundle: DatasetBundle) -> np.ndarray:
        """Predict positive-class probabilities for the test partition.

        Args:
            bundle: Dataset bundle containing test features.

        Returns:
            Positive-class probabilities.

        Raises:
            RuntimeError: If the model has not been trained.
        """

        self._validate_trained()

        probabilities = self.model.predict_proba(bundle.X_test)

        return probabilities[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        """Return coefficient importance sorted by absolute magnitude.

        Returns:
            DataFrame containing feature names, coefficients, and importance.

        Raises:
            RuntimeError: If the model has not been trained.
        """

        self._validate_trained()

        coefficients = self.model.coef_[0]
        importance = pd.DataFrame(
            {
                "feature": self._feature_names,
                "coefficient": coefficients,
                "importance": np.abs(coefficients),
            }
        )

        return importance.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

    def get_parameters(self) -> dict[str, object]:
        """Return sklearn Logistic Regression parameters.

        Returns:
            Model parameter mapping.
        """

        return self.model.get_params()

    def _validate_trained(self) -> None:
        """Ensure the model has been trained.

        Raises:
            RuntimeError: If the model has not been trained.
        """

        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction.")
