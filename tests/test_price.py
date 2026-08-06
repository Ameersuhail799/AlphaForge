from tests.utils import load_test_data

from src.features.feature_pipeline import FeaturePipeline


def main():

    df = load_test_data()

    pipeline = FeaturePipeline()

    df = pipeline.generate(df)

    print()
    print("=" * 60)
    print("PRICE FEATURES SUCCESS")
    print("=" * 60)
    print()

    columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "GAP_PCT",
        "OPEN_CLOSE_PCT",
        "HIGH_LOW_PCT",
        "BODY_SIZE",
        "UPPER_WICK",
        "LOWER_WICK",
    ]

    print(df[columns].tail())


if __name__ == "__main__":
    main()
