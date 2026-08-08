"""Append-only experiment history maintenance for AlphaForge research."""

from __future__ import annotations

import json

import pandas as pd

from config.settings import (
    EXPERIMENT_HISTORY_REPORT_PATH,
    EXPERIMENT_REPORT_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentHistory:
    """Maintain an append-only history of distinct saved experiments."""

    COLUMNS = [
        "Experiment ID",
        "Timestamp",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "Training Time",
        "Prediction Time",
        "Feature Count",
    ]

    def append_experiments(self) -> pd.DataFrame:
        """Append unrecorded experiment JSON files to the history report.

        Returns:
            Full append-only experiment history.
        """

        logger.info("Updating experiment history...")

        history = self._load_history()
        known_ids = set(history["Experiment ID"].astype(str))
        rows = [
            row
            for row in self._load_experiment_rows()
            if row["Experiment ID"] not in known_ids
        ]

        if rows:
            history = pd.concat(
                [history, pd.DataFrame(rows, columns=self.COLUMNS)],
                ignore_index=True,
            )
            EXPERIMENT_HISTORY_REPORT_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            history.to_csv(EXPERIMENT_HISTORY_REPORT_PATH, index=False)

        logger.info("Experiment history updated successfully.")

        return history

    def _load_history(self) -> pd.DataFrame:
        """Load the existing experiment history or create an empty table.

        Returns:
            Existing or empty experiment history table.
        """

        if EXPERIMENT_HISTORY_REPORT_PATH.exists():
            return pd.read_csv(EXPERIMENT_HISTORY_REPORT_PATH)

        return pd.DataFrame(columns=self.COLUMNS)

    def _load_experiment_rows(self) -> list[dict[str, object]]:
        """Read saved experiment JSON files into history rows.

        Returns:
            Serializable experiment history records.
        """

        rows: list[dict[str, object]] = []

        for report_path in sorted(EXPERIMENT_REPORT_DIR.glob("*.json")):
            with report_path.open(encoding="utf-8") as file:
                experiment = json.load(file)
            # Skip files that are not individual experiment records
            if not isinstance(experiment, dict) or "experiment_id" not in experiment:
                logger.debug("Skipping non-experiment JSON: %s", report_path)
                continue

            rows.append(
                {
                    "Experiment ID": experiment["experiment_id"],
                    "Timestamp": experiment["timestamp"],
                    "Model": experiment["model"],
                    "Accuracy": experiment["accuracy"],
                    "Precision": experiment["precision"],
                    "Recall": experiment["recall"],
                    "F1": experiment["f1"],
                    "ROC-AUC": experiment["roc_auc"],
                    "Training Time": experiment.get("training_time", 0.0),
                    "Prediction Time": experiment.get("prediction_time", 0.0),
                    "Feature Count": experiment["feature_count"],
                }
            )

        return rows
