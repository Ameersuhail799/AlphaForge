from src.data.downloader import MarketDataDownloader


def main():

    downloader = MarketDataDownloader()

    df = downloader.download_market_data()

    print()
    print("=" * 60)
    print("PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 60)
    print()

    print(df.head())

    print()

    print(f"Rows : {len(df)}")
    print(f"Columns : {len(df.columns)}")


if __name__ == "__main__":
    main()