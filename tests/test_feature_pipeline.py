from src.data.storage import StorageEngine
from src.features.feature_pipeline import FeaturePipeline


def main():

    storage = StorageEngine()

    df = storage.load_dataset("RELIANCE_NS")

    pipeline = FeaturePipeline()

    df = pipeline.generate(df)

    print()

    print("=" * 60)
    print("FEATURE PIPELINE SUCCESS")
    print("=" * 60)

    print()

    print(df.tail())

    print()

    print("Columns")

    print(df.columns.tolist())


if __name__ == "__main__":
    main()