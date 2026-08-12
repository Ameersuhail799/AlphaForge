"""Feature group analysis and ablation research for AlphaForge.

This module runs chronological, expanding-window experiments per feature
group and ablation configuration without touching the final TEST period.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.storage import StorageEngine
from src.features.feature_pipeline import FeaturePipeline
from src.dataset.target import TargetBuilder
from src.dataset.scaler import FeatureScaler
from src.models.registry import ModelRegistry
from src.models.trainer import Trainer
from src.models.evaluator import Evaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


FEATURE_GROUPS: Dict[str, List[str]] = {
    "trend": ["SMA_20", "SMA_50", "EMA_20", "EMA_50"],
    "momentum": [
        "DAILY_RETURN",
        "PRICE_CHANGE_PCT",
        "MOMENTUM_10",
        "MOMENTUM_20",
        "ROC_12",
        "RSI_14",
    ],
    "volatility": ["TRUE_RANGE", "ATR_14", "HIST_VOL_20", "ROLLING_STD_20", "DAILY_RANGE_PCT"],
    "volume": ["VOLUME_SMA_20", "VOLUME_EMA_20", "VOLUME_RATIO", "VOLUME_CHANGE_PCT", "OBV"],
    "price": ["GAP_PCT", "OPEN_CLOSE_PCT", "HIGH_LOW_PCT", "BODY_SIZE", "UPPER_WICK", "LOWER_WICK"],
}


@dataclass
class ExperimentRow:
    feature_config: str
    feature_count: int
    model: str
    fold: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    training_time: float
    prediction_time: float


def _create_folds_index(non_test_index: pd.Index, folds: int) -> list[tuple[int, int]]:
    total = len(non_test_index)
    val_size = max(1, total // (folds + 1))
    initial_train = total - folds * val_size
    positions: list[tuple[int, int]] = []
    for i in range(folds):
        train_end_pos = initial_train + i * val_size - 1
        val_end_pos = train_end_pos + val_size
        positions.append((train_end_pos, val_end_pos))
    positions = [p for p in positions if 0 <= p[0] < total and 0 <= p[1] < total]
    return positions


def _evaluate_baselines(y_train: pd.Series, y_val: pd.Series) -> Dict[str, float]:
    # Majority-class baseline: predict most frequent class in training
    majority = int(y_train.mode().iloc[0]) if not y_train.mode().empty else 0
    maj_preds = np.full(len(y_val), majority, dtype=int)

    prev_preds = y_val.shift(1).ffill().fillna(majority).astype(int).to_numpy()

    evaluator = Evaluator()
    maj_metrics = evaluator.evaluate_dummy(y_val.to_numpy(), maj_preds)
    prev_metrics = evaluator.evaluate_dummy(y_val.to_numpy(), prev_preds)

    return {"majority": maj_metrics, "previous_day": prev_metrics}


# Extend Evaluator with a small helper via monkeypatch-like function call approach
def _evaluator_evaluate_dummy(self, y_true: np.ndarray, preds: np.ndarray):
    from src.models.metrics import ModelMetrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    # ROC-AUC not meaningful without probabilities; set to None
    roc = None
    return ModelMetrics(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=roc, confusion_matrix=[], classification_report={})


# attach helper
Evaluator.evaluate_dummy = _evaluator_evaluate_dummy


def run_feature_group_experiments(
    dataset_name: str = "reliance_ns",
    folds: int = 5,
    scale: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run experiments for feature groups and ablations.

    Returns dict of DataFrames for group comparison, ablation, stability, baselines.
    """

    storage = StorageEngine()
    if not storage.dataset_exists(dataset_name):
        raise FileNotFoundError(dataset_name)

    raw = storage.load_dataset(dataset_name)

    # Generate engineered features (do not overwrite raw)
    fp = FeaturePipeline()
    df_features = fp.generate(raw.copy())

    # Build target and drop NA
    tb = TargetBuilder()
    df_with_target = tb.build(df_features)

    total = len(df_with_target)
    test_size = max(1, int(total * 0.15))
    non_test = df_with_target.iloc[:-test_size].copy()
    test_partition = df_with_target.iloc[-test_size:].copy()

    non_test_index = non_test.index
    folds_positions = _create_folds_index(non_test_index, folds)

    registry = ModelRegistry()
    models = registry.list_models()

    trainer = Trainer()
    evaluator = Evaluator()

    group_results: List[ExperimentRow] = []
    ablation_results: List[ExperimentRow] = []
    feature_importances: List[dict] = []
    baseline_rows: List[dict] = []

    # Prepare configurations: ALL, each single group, sensible two-groups
    all_features = [c for c in df_features.columns if c not in [tb.DEFAULT_TARGET]]

    group_configs = {
        "ALL": all_features,
    }

    for g, cols in FEATURE_GROUPS.items():
        group_configs[g.upper()] = [c for c in cols if c in df_features.columns]

    # sensible two-group combos: trend+momentum, price+volume
    group_configs["TREND_MOMENTUM"] = list(set(group_configs.get("TREND", []) + group_configs.get("MOMENTUM", [])))
    group_configs["PRICE_VOLUME"] = list(set(group_configs.get("PRICE", []) + group_configs.get("VOLUME", [])))

    # Run group experiments
    for config_name, feature_cols in group_configs.items():
        logger.info("Running config %s with %d features", config_name, len(feature_cols))

        for fold_idx, (train_end_pos, val_end_pos) in enumerate(folds_positions, start=1):
            train_end_idx = non_test_index[train_end_pos]
            val_start_idx = non_test_index[train_end_pos + 1]
            val_end_idx = non_test_index[val_end_pos]

            train_df = non_test.loc[:train_end_idx]
            val_df = non_test.loc[val_start_idx:val_end_idx]

            # Replace infinities and drop rows with NaNs in selected features or target to mirror DatasetBuilder behavior
            train_df = train_df.replace([np.inf, -np.inf], np.nan)
            val_df = val_df.replace([np.inf, -np.inf], np.nan)
            required = list(feature_cols) + [tb.DEFAULT_TARGET]
            train_df = train_df.dropna(subset=required).copy()
            val_df = val_df.dropna(subset=required).copy()

            # Skip fold if no usable rows remain
            if train_df.empty or val_df.empty:
                logger.info("Skipping config %s fold %d due to empty partition after dropna.", config_name, fold_idx)
                continue

            # Build X/y using only selected features
            X_train = train_df[feature_cols].copy()
            y_train = train_df[tb.DEFAULT_TARGET]
            X_val = val_df[feature_cols].copy()
            y_val = val_df[tb.DEFAULT_TARGET]

            for model_name in models:
                model = registry.create(model_name)

                scaler = FeatureScaler(scale=scale)
                X_train_scaled = scaler.fit_transform_train(X_train)
                X_val_scaled = scaler.transform(X_val)

                start_train = perf_counter()
                trainer.train(
                    model,
                    type("B", (), {"X_train": X_train_scaled, "y_train": y_train, "feature_names": feature_cols})(),
                )
                training_time = perf_counter() - start_train

                start_pred = perf_counter()
                # create temporary bundle-like object
                bundle = type("Bundle", (), {"X_test": X_val_scaled, "y_test": y_val, "feature_names": feature_cols})()
                preds = model.predict(bundle)
                probs = model.predict_proba(bundle)
                prediction_time = perf_counter() - start_pred

                metrics = evaluator.evaluate(bundle, preds, probs)

                group_results.append(
                    ExperimentRow(
                        feature_config=config_name,
                        feature_count=len(feature_cols),
                        model=model_name,
                        fold=fold_idx,
                        accuracy=metrics.accuracy,
                        precision=metrics.precision,
                        recall=metrics.recall,
                        f1=metrics.f1,
                        roc_auc=metrics.roc_auc,
                        training_time=training_time,
                        prediction_time=prediction_time,
                    )
                )

                # feature importance
                try:
                    fi = model.feature_importance()
                    for r in fi.to_dict("records"):
                        feature_importances.append({"feature": r.get("feature"), "importance": r.get("importance"), "model": model_name, "fold": fold_idx, "config": config_name})
                except Exception:
                    pass

            # baselines per fold (once per fold)
            # majority: most frequent in training; previous-day: predict using previous day's target
            maj = int(y_train.mode().iloc[0]) if not y_train.mode().empty else 0
            maj_preds = np.full(len(y_val), maj, dtype=int)
            prev_preds = y_val.shift(1).ffill().fillna(maj).astype(int).to_numpy()

            maj_metrics = evaluator.evaluate_dummy(y_val.to_numpy(), maj_preds)
            prev_metrics = evaluator.evaluate_dummy(y_val.to_numpy(), prev_preds)

            baseline_rows.append({
                "feature_config": config_name,
                "fold": fold_idx,
                "baseline": "majority",
                "accuracy": maj_metrics.accuracy,
                "precision": maj_metrics.precision,
                "recall": maj_metrics.recall,
                "f1": maj_metrics.f1,
                "roc_auc": maj_metrics.roc_auc,
            })

            baseline_rows.append({
                "feature_config": config_name,
                "fold": fold_idx,
                "baseline": "previous_day",
                "accuracy": prev_metrics.accuracy,
                "precision": prev_metrics.precision,
                "recall": prev_metrics.recall,
                "f1": prev_metrics.f1,
                "roc_auc": prev_metrics.roc_auc,
            })

    # Ablation: ALL except each group
    for gname in FEATURE_GROUPS.keys():
        excluded = FEATURE_GROUPS[gname]
        cols = [c for c in all_features if c not in excluded]
        config_name = f"ALL_EXCEPT_{gname.upper()}"

        for fold_idx, (train_end_pos, val_end_pos) in enumerate(folds_positions, start=1):
            train_end_idx = non_test_index[train_end_pos]
            val_start_idx = non_test_index[train_end_pos + 1]
            val_end_idx = non_test_index[val_end_pos]

            train_df = non_test.loc[:train_end_idx]
            val_df = non_test.loc[val_start_idx:val_end_idx]

            train_df = train_df.replace([np.inf, -np.inf], np.nan)
            val_df = val_df.replace([np.inf, -np.inf], np.nan)
            required = list(cols) + [tb.DEFAULT_TARGET]
            train_df = train_df.dropna(subset=required).copy()
            val_df = val_df.dropna(subset=required).copy()

            if train_df.empty or val_df.empty:
                logger.info("Skipping ablation %s fold %d due to empty partition after dropna.", config_name, fold_idx)
                continue

            X_train = train_df[cols].copy()
            y_train = train_df[tb.DEFAULT_TARGET]
            X_val = val_df[cols].copy()
            y_val = val_df[tb.DEFAULT_TARGET]

            for model_name in models:
                model = registry.create(model_name)
                scaler = FeatureScaler(scale=scale)
                X_train_scaled = scaler.fit_transform_train(X_train)
                X_val_scaled = scaler.transform(X_val)

                start_train = perf_counter()
                trainer.train(
                    model,
                    type("B", (), {"X_train": X_train_scaled, "y_train": y_train, "feature_names": cols})(),
                )
                training_time = perf_counter() - start_train

                bundle = type("Bundle", (), {"X_test": X_val_scaled, "y_test": y_val, "feature_names": cols})()
                start_pred = perf_counter()
                preds = model.predict(bundle)
                probs = model.predict_proba(bundle)
                prediction_time = perf_counter() - start_pred

                metrics = evaluator.evaluate(bundle, preds, probs)

                ablation_results.append(
                    ExperimentRow(
                        feature_config=config_name,
                        feature_count=len(cols),
                        model=model_name,
                        fold=fold_idx,
                        accuracy=metrics.accuracy,
                        precision=metrics.precision,
                        recall=metrics.recall,
                        f1=metrics.f1,
                        roc_auc=metrics.roc_auc,
                        training_time=training_time,
                        prediction_time=prediction_time,
                    )
                )

                try:
                    fi = model.feature_importance()
                    for r in fi.to_dict("records"):
                        feature_importances.append({"feature": r.get("feature"), "importance": r.get("importance"), "model": model_name, "fold": fold_idx, "config": config_name})
                except Exception:
                    pass

    # Persist reports
    research_dir = Path("reports") / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    df_group = pd.DataFrame([r.__dict__ for r in group_results])
    df_ablation = pd.DataFrame([r.__dict__ for r in ablation_results])
    df_fi = pd.DataFrame(feature_importances)
    df_baseline = pd.DataFrame(baseline_rows)

    df_group.to_csv(research_dir / "feature_group_comparison.csv", index=False)
    df_ablation.to_csv(research_dir / "feature_ablation_comparison.csv", index=False)
    df_fi.to_csv(research_dir / "feature_stability.csv", index=False)
    df_baseline.to_csv(research_dir / "baseline_comparison.csv", index=False)

    # Simple report
    report_lines = ["# Signal Research Report", ""]
    report_lines.append(f"Dataset: {dataset_name}")
    report_lines.append(f"Rows: {total}")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"Feature groups tested: {', '.join(sorted(FEATURE_GROUPS.keys()))}")
    (research_dir / "SIGNAL_RESEARCH_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "group": df_group,
        "ablation": df_ablation,
        "feature_stability": df_fi,
        "baseline": df_baseline,
    }
