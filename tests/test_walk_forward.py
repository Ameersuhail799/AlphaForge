"""Unit tests for the walk-forward module using deterministic synthetic data."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.research.walk_forward import _create_folds_index, run_walk_forward
from src.data.storage import StorageEngine


def _make_synthetic_dataset(rows: int = 100) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    df = pd.DataFrame(index=index)
    rng = np.random.RandomState(42)
    df["Close"] = 100 + np.cumsum(rng.normal(scale=1.0, size=rows))
    # create 5 simple features
    for i in range(5):
        df[f"f{i}"] = rng.normal(size=rows) + 0.01 * np.arange(rows)

    return df


def main():
    # basic fold index check
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    positions = _create_folds_index(idx, folds=5)
    assert len(positions) >= 1

    # create dataset and save into project raw storage
    df = _make_synthetic_dataset(120)
    storage = StorageEngine()
    storage.save_dataset(df, "test_reliance_ns", overwrite=True)

    # run walk-forward with a few folds
    result = run_walk_forward(dataset_name="test_reliance_ns", folds=3, scale=False)

    folds = result["folds"]
    summary = result["summary"]

    # checks
    assert not folds.empty
    assert "model" in folds.columns
    assert summary["fold_count"].sum() >= 1

    print()
    print("=" * 60)
    print("WALK-FORWARD TEST SUCCESS")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
