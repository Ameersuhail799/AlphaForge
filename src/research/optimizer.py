"""Hyperparameter optimization framework for AlphaForge research."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from config.settings import CHAMPION_REPORT_PATH
from src.dataset.bundle import DatasetBundle
from src.models.base_model import BaseModel
from src.models.evaluator import Evaluator
from src.models.predictor import Predictor
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
from src.research.experiment_history import ExperimentHistory
from src.research.champion import ChampionManager
from src.research.optimization_report import (
    OptimizationReportResult,
    OptimizationReportWriter,
)
from src.research.parameter_space import ParameterSpace
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OptimizationResult:
    """Contain the optimized model, parameters, and search history."""

    best_model: BaseModel
    best_parameters: dict[str, object]
    best_score: float
    history: pd.DataFrame
    report: OptimizationReportResult
    champion_updated: bool


class HyperparameterOptimizer:
    """Optimize registered AlphaForge models with time-series CV."""

    def __init__(self, n_iter: int = 10, random_state: int = 42) -> None:
        """Initialize the optimizer.

        Args:
            n_iter: Number of random search samples.
            random_state: Sampling seed used by RandomizedSearchCV.
        """

        self.n_iter = n_iter
        self.random_state = random_state
        self.registry = ModelRegistry()

    def optimize(self, model_name: str, bundle: DatasetBundle) -> OptimizationResult:
        """Optimize a registered model for a dataset bundle.

        Args:
            model_name: Registered AlphaForge model name.
            bundle: Dataset bundle to use for training and validation.

        Returns:
            Best model, parameters, search history, and report metadata.

        Raises:
            ValueError: If the model is not supported.
        """

        logger.info("Starting hyperparameter optimization for %s...", model_name)

        model = self.registry.create(model_name)
        search_start = perf_counter()
        search = self._search(model, bundle)
        search_duration = perf_counter() - search_start

        # Build and train the best estimator on the TRAINING split only
        best_model = self.registry.create(model_name, **search.best_params_)
        trainer = Trainer()
        trainer.train(best_model, bundle)

        # Evaluate on VALIDATION split (must not use TEST for selection)
        # Use underlying sklearn estimator for direct predictions on X_valid
        if not hasattr(best_model, "model"):
            raise RuntimeError("Optimized model must expose underlying sklearn estimator")

        estimator = getattr(best_model, "model")
        import numpy as _np

        val_preds = estimator.predict(bundle.X_valid)
        val_probs = estimator.predict_proba(bundle.X_valid)[:, 1]

        # Evaluator expects a bundle-like object with y_test attribute
        validation_metrics = Evaluator().evaluate(
            type("B", (), {"y_test": bundle.y_valid}),
            _np.asarray(val_preds),
            _np.asarray(val_probs),
        )

        # Decide champion using VALIDATION metrics only
        champion_updated = self._update_champion_if_improved_on_validation(
            model_name, search.best_params_, validation_metrics
        )

        # Retrain on TRAIN + VALID before final test evaluation
        X_combined = pd.concat([bundle.X_train, bundle.X_valid])
        y_combined = pd.concat([bundle.y_train, bundle.y_valid])

        combined_bundle = DatasetBundle(
            X_train=X_combined,
            y_train=y_combined,
            X_valid=bundle.X_valid,
            y_valid=bundle.y_valid,
            X_test=bundle.X_test,
            y_test=bundle.y_test,
            feature_names=bundle.feature_names,
            target_name=bundle.target_name,
            train_start=bundle.train_start,
            train_end=bundle.validation_end,
            validation_start=bundle.validation_start,
            validation_end=bundle.validation_end,
            test_start=bundle.test_start,
            test_end=bundle.test_end,
            metadata=bundle.metadata,
        )

        trainer.train(best_model, combined_bundle)

        # Final evaluation on TEST split
        test_preds = estimator.predict(bundle.X_test)
        test_probs = estimator.predict_proba(bundle.X_test)[:, 1]

        test_metrics = Evaluator().evaluate(
            type("B", (), {"y_test": bundle.y_test}),
            _np.asarray(test_preds),
            _np.asarray(test_probs),
        )

        # Persist experiment and update comparison/history using TEST metrics
        trainer.save_experiment(
            best_model,
            bundle,
            test_metrics,
            experiment_id=f"{model_name}_optimization",
        )
        ExperimentHistory().append_experiments()

        report = OptimizationReportWriter().write(
            model_name=model_name,
            best_parameters=search.best_params_,
            best_score=float(search.best_score_),
            history=self._history_frame(search),
            execution_time_seconds=search_duration,
            validation_score=validation_metrics.to_dict(),
            final_test_score=test_metrics.to_dict(),
        )

        logger.info("Hyperparameter optimization completed for %s.", model_name)

        return OptimizationResult(
            best_model=best_model,
            best_parameters=search.best_params_,
            best_score=float(search.best_score_),
            history=self._history_frame(search),
            report=report,
            champion_updated=champion_updated,
        )

    def _search(self, model: BaseModel, bundle: DatasetBundle) -> RandomizedSearchCV:
        """Run randomized search with chronological cross-validation.

        Args:
            model: Base model instance.
            bundle: Dataset bundle to optimize against.

        Returns:
            Fitted randomized search object.
        """

        estimator = clone(model.model)
        search_space = ParameterSpace.get(model.model_name)
        cv = TimeSeriesSplit(n_splits=5)

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=search_space,
            n_iter=self.n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            refit=True,
        )

        search.fit(bundle.X_train, bundle.y_train)

        return search

    def _history_frame(self, search: RandomizedSearchCV) -> pd.DataFrame:
        """Convert search results into a deterministic history table.

        Args:
            search: Completed randomized search.

        Returns:
            Sorted search history records.
        """

        history = pd.DataFrame(search.cv_results_)
        history = history[
            [
                "params",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
                "mean_fit_time",
                "mean_score_time",
            ]
        ].copy()
        history["params"] = history["params"].apply(self._sorted_mapping)

        return history.sort_values(
            ["rank_test_score", "mean_test_score"],
            ascending=[True, False],
            kind="mergesort",
        ).reset_index(drop=True)

    def _sorted_mapping(self, parameters: dict[str, object]) -> dict[str, object]:
        """Return a mapping with stable key ordering for reporting.

        Args:
            parameters: Search parameters.

        Returns:
            Sorted parameter dictionary.
        """

        return dict(sorted(parameters.items(), key=lambda item: item[0]))

    def _update_champion_if_improved(
        self,
        model: BaseModel,
        bundle: DatasetBundle,
    ) -> bool:
        """Update champion artifacts only when the optimized model improves.

        Args:
            model: Optimized model.
            bundle: Dataset bundle used for scoring.

        Returns:
            ``True`` when the champion artifacts were refreshed.
        """

        current_champion = None
        if CHAMPION_REPORT_PATH.exists():
            current_champion = pd.read_json(CHAMPION_REPORT_PATH, typ="series")

        if current_champion is not None:
            current_rank = (
                float(current_champion["roc_auc"]),
                float(current_champion["f1"]),
                float(current_champion["accuracy"]),
            )
        else:
            current_rank = (float("-inf"), float("-inf"), float("-inf"))

        predictor = Predictor()
        predictions = predictor.predict(model, bundle)
        probabilities = predictor.predict_probabilities(model, bundle)
        metrics = Evaluator().evaluate(bundle, predictions.to_numpy(), probabilities.to_numpy())

        new_rank = (
            float(metrics.roc_auc) if metrics.roc_auc is not None else float("-inf"),
            float(metrics.f1),
            float(metrics.accuracy),
        )

        if new_rank <= current_rank:
            return False

        ChampionManager().generate()
        return True

    def _update_champion_if_improved_on_validation(
        self,
        model_name: str,
        parameters: dict[str, object],
        validation_metrics,
    ) -> bool:
        """Update champion using VALIDATION metrics only.

        This writes `reports/champion.json` with validation metrics and
        the selected parameters when the new model improves over the
        current champion according to ROC-AUC, then F1, then Accuracy.
        """

        current = None
        if CHAMPION_REPORT_PATH.exists():
            try:
                import json as _json

                current = _json.loads(CHAMPION_REPORT_PATH.read_text(encoding="utf-8"))
            except Exception:
                current = None

        def _key_from_current(d: dict[str, object] | None) -> tuple[float, float, float]:
            if not d:
                return (float("-inf"), float("-inf"), float("-inf"))

            return (
                float(d.get("validation_roc_auc") or d.get("roc_auc") or float("-inf")),
                float(d.get("validation_f1") or d.get("f1") or float("-inf")),
                float(d.get("validation_accuracy") or d.get("accuracy") or float("-inf")),
            )

        current_key = _key_from_current(current)

        new_key = (
            float(validation_metrics.roc_auc) if validation_metrics.roc_auc is not None else float("-inf"),
            float(validation_metrics.f1),
            float(validation_metrics.accuracy),
        )

        # Preserve existing champion on exact ties
        if new_key <= current_key:
            return False

        # Write champion.json with validation metrics (test metrics are written later)
        from datetime import datetime, timezone

        payload = {
            "model": model_name,
            "accuracy": float(validation_metrics.accuracy),
            "precision": float(validation_metrics.precision),
            "recall": float(validation_metrics.recall),
            "f1": float(validation_metrics.f1),
            "roc_auc": float(validation_metrics.roc_auc) if validation_metrics.roc_auc is not None else None,
            "parameters": parameters,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "Improved validation ROC-AUC",
        }

        CHAMPION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHAMPION_REPORT_PATH.write_text(__import__("json").dumps(payload, indent=4), encoding="utf-8")

        return True