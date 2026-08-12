"""Tests for feature research module."""

from __future__ import annotations

import os

from src.research.feature_research import FEATURE_GROUPS, run_feature_group_experiments


def main():
    # Basic sanity run (small number of folds to keep fast)
    res = run_feature_group_experiments(dataset_name="reliance_ns", folds=3, scale=False)

    assert "group" in res
    assert "ablation" in res
    assert "feature_stability" in res
    assert "baseline" in res

    # Check feature groups exist
    assert "trend" in FEATURE_GROUPS
    assert "momentum" in FEATURE_GROUPS

    # Check reports were written
    assert os.path.exists("reports/research/feature_group_comparison.csv")
    assert os.path.exists("reports/research/feature_ablation_comparison.csv")
    assert os.path.exists("reports/research/feature_stability.csv")
    assert os.path.exists("reports/research/baseline_comparison.csv")

    print("FEATURE RESEARCH TESTS PASSED")


if __name__ == "__main__":
    main()
