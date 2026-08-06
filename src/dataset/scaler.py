"""Feature scaling for AlphaForge datasets."""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureScaler:
    """Optionally standardize features using training data statistics only."""

    def __init__(self, scale: bool = False) -> None:
        """Initialize the feature scaler.

        Args:
            scale: Whether standard scaling should be applied.
        """

        self.scale = scale
        self._mean: pd.Series | None = None
        self._scale: pd.Series | None = None

    def fit_transform_train(
        self,
        X_train: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit scaling statistics on training data and transform it.

        Args:
            X_train: Training feature matrix.

        Returns:
            Unchanged training data when disabled, otherwise standardized data.
        """

        if not self.scale:
            return X_train

        logger.info("Fitting StandardScaler on training features...")

        self._mean = X_train.mean()
        self._scale = X_train.std(ddof=0).replace(0, 1.0)

        return self.transform(X_train)

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform data with statistics fitted on the training partition.

        Args:
            X: Feature matrix to transform.

        Returns:
            Unchanged data when disabled, otherwise standardized data.

        Raises:
            RuntimeError: If scaling is enabled before fitting on training data.
        """

        if not self.scale:
            return X

        if self._mean is None or self._scale is None:
            raise RuntimeError("FeatureScaler must be fit before transforming.")

        logger.info("Transforming feature partition with training statistics...")

        return (X - self._mean) / self._scale
