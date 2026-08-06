from tests.utils import load_test_data

from src.features.feature_pipeline import FeaturePipeline


def main():

    df = load_test_data()

    pipeline = FeaturePipeline()

    df = pipeline.generate(df)

    print()
    print("=" * 60)
    print("MOMENTUM FEATURES SUCCESS")
    print("=" * 60)
    print()

    columns = [
        "Close",
        "RSI_14",
        "ROC_12",
        "MOMENTUM_10",
        "MOMENTUM_20",
        "DAILY_RETURN",
        "PRICE_CHANGE_PCT",
    ]

    print(df[columns].tail())


if __name__ == "__main__":
    main()