"""Regression test for ExperimentHistory JSON classification.

This test ensures ExperimentHistory loads valid single-experiment JSON
files and ignores research/aggregate JSON files that lack `experiment_id`.

It writes temporary JSON files into the existing `reports/experiments/`
directory (without moving or deleting existing files), runs
`ExperimentHistory.append_experiments()`, and then restores any
preexisting history CSV.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from src.research.experiment_history import ExperimentHistory
from config.settings import EXPERIMENT_REPORT_DIR, EXPERIMENT_HISTORY_REPORT_PATH


def test_experiment_history_handles_mixed_json_files():
    EXPERIMENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    valid_path = EXPERIMENT_REPORT_DIR / "valid_experiment_unit_test.json"
    research_path = EXPERIMENT_REPORT_DIR / "research_aggregate_unit_test.json"

    # Backup existing history file if present
    backup_history = None
    if EXPERIMENT_HISTORY_REPORT_PATH.exists():
        backup_history = EXPERIMENT_HISTORY_REPORT_PATH.read_bytes()

    try:
        # Write a valid single-experiment JSON
        valid = {
            "experiment_id": "valid_experiment_unit_test",
            "timestamp": datetime.utcnow().isoformat() + "+00:00",
            "model": "unit_test_model",
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "roc_auc": 0.5,
            "feature_count": 3,
            "training_time": 0.01,
            "prediction_time": 0.0,
        }

        # Write a research/aggregate JSON that should be ignored
        research = {"models": ["a", "b", "c"], "summary": {"note": "aggregate"}}

        valid_path.write_text(json.dumps(valid), encoding="utf-8")
        research_path.write_text(json.dumps(research), encoding="utf-8")

        # Run the history append operation — should not raise and should include the valid id
        history = ExperimentHistory().append_experiments()

        assert "valid_experiment_unit_test" in history["Experiment ID"].astype(str).values

    finally:
        # Cleanup test files
        try:
            valid_path.unlink()
        except FileNotFoundError:
            pass
        try:
            research_path.unlink()
        except FileNotFoundError:
            pass

        # Restore original history CSV if it existed, otherwise remove created one
        if backup_history is not None:
            EXPERIMENT_HISTORY_REPORT_PATH.write_bytes(backup_history)
        else:
            try:
                EXPERIMENT_HISTORY_REPORT_PATH.unlink()
            except FileNotFoundError:
                pass
