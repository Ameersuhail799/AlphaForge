from tests.utils import load_test_data

from src.dataset.builder import DatasetBuilder
from src.dataset.bundle import DatasetBundle
from src.features.feature_pipeline import FeaturePipeline


def main():

    df = load_test_data()

    pipeline = FeaturePipeline()
    engineered_df = pipeline.generate(df)

    builder = DatasetBuilder(scale=True)
    bundle = builder.build(engineered_df, symbol="RELIANCE.NS")

    total_rows = len(
        bundle.X_train
    ) + len(bundle.X_valid) + len(bundle.X_test)

    assert isinstance(bundle, DatasetBundle)
    assert bundle.target_name == "NEXT_DAY_DIRECTION"
    assert total_rows == bundle.metadata["rows"]
    assert not bundle.X_train.isna().any().any()
    assert not bundle.X_valid.isna().any().any()
    assert not bundle.X_test.isna().any().any()
    assert not bundle.y_train.isna().any()
    assert not bundle.y_valid.isna().any()
    assert not bundle.y_test.isna().any()
    assert len(bundle.X_train) == int(total_rows * 0.70)
    assert len(bundle.X_valid) == int(total_rows * 0.15)
    assert len(bundle.X_test) == total_rows - (
        len(bundle.X_train) + len(bundle.X_valid)
    )
    assert bundle.train_end < bundle.validation_start
    assert bundle.validation_end < bundle.test_start
    assert abs(bundle.X_train.mean().abs().max()) < 1e-10
    assert bundle.metadata["scaled"] is True

    print()
    print("=" * 60)
    print("DATASET BUILDER SUCCESS")
    print("=" * 60)
    print()
    print(bundle.metadata)


if __name__ == "__main__":
    main()
