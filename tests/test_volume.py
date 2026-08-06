from tests.utils import load_test_data

from src.features.feature_pipeline import FeaturePipeline


def main():

    df = load_test_data()

    pipeline = FeaturePipeline()

    df = pipeline.generate(df)

    print()
    print("=" * 60)
    print("VOLUME FEATURES SUCCESS")
    print("=" * 60)
    print()

    columns = [
        "Close",
        "Volume",
        "VOLUME_SMA_20",
        "VOLUME_EMA_20",
        "VOLUME_RATIO",
        "VOLUME_CHANGE_PCT",
        "OBV",
    ]

    print(df[columns].tail())


if __name__ == "__main__":
    main()
