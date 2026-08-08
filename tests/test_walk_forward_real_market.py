"""Run the walk-forward experiment against the stored RELIANCE_NS dataset."""

from __future__ import annotations

from src.research.walk_forward import run_walk_forward


def main():
    result = run_walk_forward(dataset_name="reliance_ns", folds=5, scale=False)

    folds = result["folds"]
    summary = result["summary"]

    assert not folds.empty
    assert "model" in folds.columns

    print()
    print("=" * 60)
    print("WALK-FORWARD REAL-MARKET SUCCESS")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
