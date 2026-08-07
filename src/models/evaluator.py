"""Model evaluation for AlphaForge classification workflows."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.dataset.bundle import DatasetBundle
from src.models.metrics import ModelMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Calculate classification metrics against a dataset bundle test split."""

    def evaluate(
        self,
        bundle: DatasetBundle,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> ModelMetrics:
        """Evaluate test predictions and probabilities.

        Args:
            bundle: Dataset bundle containing test targets.
            predictions: Predicted target classes.
            probabilities: Positive-class probabilities.

        Returns:
            Classification metric results.
        """

        logger.info("Evaluating model predictions...")

        y_test = bundle.y_test
        roc_auc = self._calculate_roc_auc(y_test, probabilities)

        metrics = ModelMetrics(
            accuracy=float(accuracy_score(y_test, predictions)),
            precision=float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            recall=float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            f1=float(f1_score(y_test, predictions, zero_division=0)),
            roc_auc=roc_auc,
            confusion_matrix=confusion_matrix(
                y_test,
                predictions,
            ).tolist(),
            classification_report=classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        )

        logger.info("Model evaluation completed successfully.")

        return metrics

    def _calculate_roc_auc(
        self,
        y_test: np.ndarray,
        probabilities: np.ndarray,
    ) -> float | None:
        """Calculate ROC-AUC when both test classes are represented.

        Args:
            y_test: Ground-truth test targets.
            probabilities: Positive-class probabilities.

        Returns:
            ROC-AUC, or None when the test partition has one class.
        """

        if len(np.unique(y_test)) < 2:
            logger.warning(
                "ROC-AUC is unavailable because the test split has one class."
            )
            return None

        return float(roc_auc_score(y_test, probabilities))
