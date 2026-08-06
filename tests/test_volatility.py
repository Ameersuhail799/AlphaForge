from tests.utils import load_test_data

from src.features.feature_pipeline import FeaturePipeline


def main():

    df = load_test_data()

    pipeline = FeaturePipeline()

    df = pipeline.generate(df)

    print()
    print("=" * 60)
    print("VOLATILITY FEATURES SUCCESS")
    print("=" * 60)
    print()

    columns = [
        "Close",
        "TRUE_RANGE",
        "ATR_14",
        "HIST_VOL_20",
        "ROLLING_STD_20",
        "DAILY_RANGE_PCT",
    ]

    print(df[columns].tail())


if __name__ == "__main__":
    main()