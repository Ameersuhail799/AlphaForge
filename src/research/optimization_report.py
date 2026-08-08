"""Optimization report generation for AlphaForge research."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config.settings import REPORT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OptimizationReportResult:
    """Contain optimization report data and output path."""

    report: dict[str, object]
    report_path: Path


class OptimizationReportWriter:
    """Persist hyperparameter optimization results as JSON."""

    def write(
        self,
        model_name: str,
        best_parameters: dict[str, object],
        best_score: float,
        history: pd.DataFrame,
        execution_time_seconds: float,
        validation_score: dict[str, object] | None = None,
        final_test_score: dict[str, object] | None = None,
    ) -> OptimizationReportResult:
        """Write a deterministic optimization report.

        Args:
            model_name: Optimized model name.
            best_parameters: Best parameter combination.
            best_score: Best cross-validation score.
            history: Full evaluation history.
            execution_time_seconds: Optimization runtime.

        Returns:
            Report content and saved path.
        """

        logger.info("Writing optimization report for %s...", model_name)

        report_dir = REPORT_DIR / "optimization"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{model_name}_optimization.json"

        report = {
            "model": model_name,
            "best_parameters": best_parameters,
            "optimization_metric": best_score,
            "evaluated_combinations": history.to_dict(orient="records"),
            "execution_time_seconds": execution_time_seconds,
            "validation_score": validation_score,
            "final_test_score": final_test_score,
        }

        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

        logger.info("Optimization report saved to %s.", report_path)

        return OptimizationReportResult(report=report, report_path=report_path)