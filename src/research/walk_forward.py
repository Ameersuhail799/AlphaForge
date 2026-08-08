"""Walk-forward expanding-window validation utilities for AlphaForge.

Provides a reusable function to run expanding-window, chronological
walk-forward experiments without touching the final holdout TEST period.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable
from pathlib import Path

import pandas as pd

from src.data.storage import StorageEngine
from src.dataset.target import TargetBuilder
from src.dataset.scaler import FeatureScaler
from src.dataset.bundle import DatasetBundle
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
from src.models.evaluator import Evaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FoldResult:
    fold_id: int
    train_start: object
    train_end: object
    validation_start: object
    validation_end: object
    train_samples: int
    validation_samples: int
    model: str
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    training_time: float
    prediction_time: float


def _create_folds_index(non_test_index: pd.Index, folds: int) -> list[tuple[int, int]]:
    """Return list of (train_end_pos, validation_end_pos) index positions for folds.

    Uses an expanding-training-window design where the non-test portion is
    partitioned into `folds` validation blocks of equal size and an initial
    training block that consumes the leftover rows.
    """

    total = len(non_test_index)

    if folds < 1:
        raise ValueError("folds must be >= 1")

    # validation block size determined by dividing non-test portion into folds+1
    val_size = max(1, total // (folds + 1))
    initial_train = total - folds * val_size

    positions: list[tuple[int, int]] = []

    for i in range(folds):
        train_end_pos = initial_train + i * val_size - 1
        val_end_pos = train_end_pos + val_size
        positions.append((train_end_pos, val_end_pos))

    # filter out any fold whose indices are out of bounds
    positions = [p for p in positions if 0 <= p[0] < total and 0 <= p[1] < total]

    return positions


def run_walk_forward(
    dataset_name: str = "reliance_ns",
    models: Iterable[str] | None = None,
    folds: int = 5,
    scale: bool = False,
    output_dir: str | None = None,
) -> dict:
    """Run an expanding-window walk-forward experiment.

    Args:
        dataset_name: Stored dataset name (parquet under raw/).
        models: Iterable of model names registered in ModelRegistry. If None,
            run the default registry list.
        folds: Desired number of validation folds (approximate).
        scale: Whether to standardize features per-fold using training stats.
        output_dir: Optional base directory for research reports (unused,
            report files are written to reports/research/ by convention).

    Returns:
        Dictionary containing fold-level results and summary statistics.
    """

    storage = StorageEngine()
    if not storage.dataset_exists(dataset_name):
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")

    df = storage.load_dataset(dataset_name)

    # Build target using existing TargetBuilder logic
    target_builder = TargetBuilder()
    df_with_target = target_builder.build(df)

    # Preserve final test partition: last 15% per DatasetBuilder convention
    total = len(df_with_target)
    test_size = max(1, int(total * 0.15))
    non_test = df_with_target.iloc[:-test_size].copy()
    test_partition = df_with_target.iloc[-test_size:].copy()

    non_test_index = non_test.index
    if len(non_test_index) < 3:
        raise ValueError("Insufficient non-test rows for walk-forward folds.")

    folds_positions = _create_folds_index(non_test_index, folds)

    registry = ModelRegistry()
    if models is None:
        models_to_run = registry.list_models()
    else:
        models_to_run = list(models)

    results: list[FoldResult] = []
    feature_importances: list[dict] = []

    trainer = Trainer()
    evaluator = Evaluator()

    for fold_idx, (train_end_pos, val_end_pos) in enumerate(folds_positions, start=1):
        train_start_idx = non_test_index[0]
        train_end_idx = non_test_index[train_end_pos]
        val_start_idx = non_test_index[train_end_pos + 1]
        val_end_idx = non_test_index[val_end_pos]

        train_df = non_test.loc[:train_end_idx]
        val_df = non_test.loc[val_start_idx:val_end_idx]

        X_train = train_df.drop(columns=[TargetBuilder.DEFAULT_TARGET])
        y_train = train_df[TargetBuilder.DEFAULT_TARGET]
        X_val = val_df.drop(columns=[TargetBuilder.DEFAULT_TARGET])
        y_val = val_df[TargetBuilder.DEFAULT_TARGET]

        for model_name in models_to_run:
            model = registry.create(model_name)

            scaler = FeatureScaler(scale=scale)
            X_train_scaled = scaler.fit_transform_train(X_train)
            X_val_scaled = scaler.transform(X_val)

            bundle = DatasetBundle(
                X_train=X_train_scaled,
                y_train=y_train,
                X_valid=pd.DataFrame(),
                y_valid=pd.Series(dtype=int),
                X_test=X_val_scaled,
                y_test=y_val,
                feature_names=X_train.columns.tolist(),
                target_name=TargetBuilder.DEFAULT_TARGET,
                train_start=train_start_idx,
                train_end=train_end_idx,
                validation_start=val_start_idx,
                validation_end=val_end_idx,
                test_start=test_partition.index[0],
                test_end=test_partition.index[-1],
                metadata={},
            )

            # Train
            start_train = perf_counter()
            trainer.train(model, bundle)
            training_time = perf_counter() - start_train

            # Predict on validation (bundle.X_test contains validation)
            start_pred = perf_counter()
            preds = model.predict(bundle)
            probs = model.predict_proba(bundle)
            prediction_time = perf_counter() - start_pred

            metrics = evaluator.evaluate(bundle, preds, probs)

            # Record fold result
            fr = FoldResult(
                fold_id=fold_idx,
                train_start=train_start_idx,
                train_end=train_end_idx,
                validation_start=val_start_idx,
                validation_end=val_end_idx,
                train_samples=len(X_train_scaled),
                validation_samples=len(X_val_scaled),
                model=model_name,
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                roc_auc=metrics.roc_auc,
                training_time=training_time,
                prediction_time=prediction_time,
            )

            results.append(fr)

            # Feature importance export
            try:
                fi = model.feature_importance()
                fi_records = fi.to_dict("records")
                for r in fi_records:
                    feature_importances.append(
                        {
                            "fold_id": fold_idx,
                            "model": model_name,
                            **r,
                        }
                    )
            except Exception:
                # Some models may not expose feature importance; skip silently.
                logger.debug("No feature importance for %s on fold %d.", model_name, fold_idx)

    # Persist results to reports/research/
    research_dir = Path("reports") / "research"

    research_dir.mkdir(parents=True, exist_ok=True)

    # fold-level CSV
    fold_rows = [fr.__dict__ for fr in results]
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(research_dir / "walk_forward_results.csv", index=False)

    # summary per model
    summary = (
        folds_df.groupby("model")
        .agg(
            fold_count=("fold_id", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            median_roc_auc=("roc_auc", "median"),
            min_roc_auc=("roc_auc", "min"),
            max_roc_auc=("roc_auc", "max"),
        )
        .reset_index()
    )

    # counts above/below 0.50
    def count_above(series: pd.Series, threshold: float) -> int:
        return int((series > threshold).sum())

    def count_below(series: pd.Series, threshold: float) -> int:
        return int((series < threshold).sum())

    rows = []
    for _, row in summary.iterrows():
        model = row["model"]
        roc_series = folds_df.loc[folds_df["model"] == model, "roc_auc"].dropna()
        rows.append(
            {
                "model": model,
                **row.to_dict(),
                "folds_above_roc_auc_0_50": count_above(roc_series, 0.5),
                "folds_below_roc_auc_0_50": count_below(roc_series, 0.5),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(research_dir / "walk_forward_summary.csv", index=False)

    fi_df = pd.DataFrame(feature_importances)
    fi_df.to_csv(research_dir / "walk_forward_feature_importance.csv", index=False)

    # simple markdown report
    report_lines = [
        "# Temporal Robustness Report",
        "",
        f"Dataset: {dataset_name}",
        f"Rows: {total}",
        "",
        "## Summary",
        "",
    ]

    for _, r in summary_df.sort_values("mean_roc_auc", ascending=False).iterrows():
        report_lines.append(
            f"- {r['model']}: mean ROC-AUC={r['mean_roc_auc']:.4f}, std={r['std_roc_auc']:.4f}"
        )

    (research_dir / "TEMPORAL_ROBUSTNESS_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "folds": folds_df,
        "summary": summary_df,
        "feature_importance": fi_df,
    }
