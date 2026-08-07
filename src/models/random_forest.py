"""Random Forest model implementation for AlphaForge."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.dataset.bundle import DatasetBundle
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """Train and serve a Random Forest classifier from a dataset bundle."""

    model_name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
        **parameters: object,
    ) -> None:
        """Initialize the sklearn Random Forest estimator.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum tree depth, or None for unlimited depth.
            random_state: Seed used by the underlying estimator.
            n_jobs: Number of parallel training workers.
            **parameters: Additional RandomForestClassifier parameters.
        """

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            **parameters,
        )
        self._feature_names: list[str] = []
        self._is_trained = False

    def fit(self, bundle: DatasetBundle) -> None:
        """Fit Random Forest on a dataset bundle's training partition.

        Args:
            bundle: Dataset bundle containing training features and targets.
        """

        logger.info("Training Random Forest model...")

        self.model.fit(bundle.X_train, bundle.y_train)
        self._feature_names = bundle.feature_names.copy()
        self._is_trained = True

        logger.info("Random Forest model trained successfully.")

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

        return self.model.predict_proba(bundle.X_test)[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance sorted from highest to lowest.

        Returns:
            DataFrame containing feature names and Random Forest importance.

        Raises:
            RuntimeError: If the model has not been trained.
        """

        self._validate_trained()

        importance = pd.DataFrame(
            {
                "feature": self._feature_names,
                "importance": self.model.feature_importances_,
            }
        )

        return importance.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

    def get_parameters(self) -> dict[str, object]:
        """Return sklearn Random Forest parameters.

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
