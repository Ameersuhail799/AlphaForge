from tests.utils import load_test_data

from src.analysis.report_generator import ReportGenerator
from src.dataset.builder import DatasetBuilder
from src.features.feature_pipeline import FeaturePipeline


def main():

    df = load_test_data()
    engineered_df = FeaturePipeline().generate(df)
    bundle = DatasetBuilder(scale=False).build(
        engineered_df,
        symbol="RELIANCE.NS",
    )

    result = ReportGenerator().generate(bundle)

    assert result.dataset_statistics.total_samples == bundle.metadata["rows"]
    assert result.dataset_statistics.feature_count == len(bundle.feature_names)
    assert result.dataset_statistics.missing_values == 0
    assert (
        result.class_distribution.class_zero_count
        + result.class_distribution.class_one_count
        == result.dataset_statistics.total_samples
    )
    assert result.feature_statistics.shape[0] == len(bundle.feature_names)
    assert {
        "mean",
        "median",
        "std",
        "min",
        "max",
        "variance",
        "skewness",
        "kurtosis",
    }.issubset(result.feature_statistics.columns)
    assert len(result.correlation_analysis.target_correlation) == len(
        bundle.feature_names
    )
    assert result.class_distribution.plot_path.exists()
    assert result.correlation_analysis.matrix_path.exists()
    assert result.correlation_analysis.target_path.exists()
    assert result.correlation_analysis.heatmap_path.exists()
    assert result.report_path.exists()

    print()
    print("=" * 60)
    print("BASELINE ANALYSIS SUCCESS")
    print("=" * 60)
    print()
    print(result.correlation_analysis.target_correlation.head())


if __name__ == "__main__":
    main()
