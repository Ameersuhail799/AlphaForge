"""XGBoost model integration for AlphaForge.

Implements the ``BaseModel`` interface using ``xgboost.XGBClassifier``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.dataset.bundle import DatasetBundle
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classifier wrapper following AlphaForge model conventions.

    This class wraps ``xgboost.XGBClassifier`` and exposes the same
    dataset-bundle-based interface used by other models in the project.
    """

    model_name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        random_state: int = 42,
        n_jobs: int = -1,
        eval_metric: str = "logloss",
        **parameters: Any,
    ) -> None:
        """Initialize the XGBoost classifier used by AlphaForge.

        Args:
            See class docstring for parameter meanings. Extra keyword
            arguments are forwarded to ``xgboost.XGBClassifier``.
        """

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            use_label_encoder=False,
            eval_metric=eval_metric,
            **parameters,
        )

        self._feature_names: list[str] = []
        self._is_trained = False

    def fit(self, bundle: DatasetBundle) -> None:
        """Train the XGBoost model using the bundle's training partition.

        Args:
            bundle: DatasetBundle with training data.
        """

        logger.info("Training XGBoost model...")

        self.model.fit(bundle.X_train, bundle.y_train)
        self._feature_names = bundle.feature_names.copy()
        self._is_trained = True

        logger.info("XGBoost model trained successfully.")

    def predict(self, bundle: DatasetBundle) -> np.ndarray:
        """Predict class labels for the bundle's test partition.

        Raises:
            RuntimeError: If the model is not trained.
        """

        self._validate_trained()

        return self.model.predict(bundle.X_test)

    def predict_proba(self, bundle: DatasetBundle) -> np.ndarray:
        """Predict positive-class probabilities for the test partition.

        Raises:
            RuntimeError: If the model is not trained.
        """

        self._validate_trained()

        return self.model.predict_proba(bundle.X_test)[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance ordered from highest to lowest.

        Returns:
            DataFrame with columns ``feature`` and ``importance``.

        Raises:
            RuntimeError: If the model is not trained.
        """

        self._validate_trained()

        importance = pd.DataFrame(
            {
                "feature": self._feature_names,
                "importance": getattr(self.model, "feature_importances_", []),
            }
        )

        return importance.sort_values("importance", ascending=False).reset_index(
            drop=True
        )

    def get_parameters(self) -> dict[str, object]:
        """Return underlying XGBoost model parameters."""

        return self.model.get_params()

    def _validate_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction.")
