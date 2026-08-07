"""Evaluation metric definitions for AlphaForge models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelMetrics:
    """Contain classification metrics and diagnostic outputs."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: list[list[int]]
    classification_report: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Convert metric results to a serializable mapping.

        Returns:
            Dictionary representation of the metrics.
        """

        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
            "classification_report": self.classification_report,
        }
