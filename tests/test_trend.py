from src.data.downloader import MarketDataDownloader
from src.features.trend import TrendFeatureGenerator


def main():

    downloader = MarketDataDownloader()

    df = downloader.download_market_data()

    generator = TrendFeatureGenerator()

    df = generator.generate(df)

    print()

    print(df[[
        "Close",
        "SMA_20",
        "SMA_50",
        "EMA_20",
        "EMA_50",
    ]].tail())

    print()

    print("Trend Feature Test Passed")


if __name__ == "__main__":
    main()