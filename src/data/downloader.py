"""
AlphaForge Market Data Downloader

Coordinates the complete data ingestion pipeline.
"""

from __future__ import annotations

import pandas as pd

from config.settings import DEFAULT_SYMBOL, START_DATE

from src.data.providers.yahoo_provider import YahooProvider
from src.data.storage import StorageEngine
from src.data.validator import DataValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataDownloader:
    """
    Complete AlphaForge market data pipeline.
    """

    def __init__(self) -> None:
        self.provider = YahooProvider()
        self.validator = DataValidator()
        self.storage = StorageEngine()

    def download_market_data(
        self,
        symbol: str = DEFAULT_SYMBOL,
        start_date: str = START_DATE,
    ):
        """
        Download, validate, save and return market data.

        Args:
            symbol: Yahoo Finance ticker.
            start_date: Historical start date.

        Returns:
            Validated pandas DataFrame.
        """

        logger.info("=" * 60)
        logger.info("Starting AlphaForge Data Pipeline")
        logger.info("=" * 60)

        # Download
        df = self.provider.download(
            symbol=symbol,
            start_date=start_date,
        )

        dataset_name = symbol.replace(".", "_")

        # Clean MultiIndex columns and missing rows prior to validation
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=[col for col in DataValidator.REQUIRED_COLUMNS if col in df.columns]).copy()

        # Validate
        report = self.validator.validate(df)

        if not report.passed:
            logger.error("Validation failed: %s", report)
            raise RuntimeError(report.message)

        logger.info("Validation passed.")

        # Save dataset
        self.storage.save_dataset(
            df=df,
            dataset_name=dataset_name,
            overwrite=True,
        )

        # Metadata
        metadata = self.storage.generate_metadata(
            df=df,
            dataset_name=dataset_name,
        )

        self.storage.save_metadata(
            dataset_name=dataset_name,
            metadata=metadata,
        )

        logger.info("Pipeline completed successfully.")

        return df


def refresh_all_datasets(start_date: str = START_DATE) -> dict[str, bool]:
    """Re-fetch OHLCV parquet files for all 5 liquid NSE equities up to the most recent completed trading day.

    Returns:
        Status dictionary mapping asset symbols to boolean success flags.
    """
    from src.production.trading_engine import SUPPORTED_ASSETS

    downloader = MarketDataDownloader()
    results = {}

    logger.info("Refreshing raw OHLCV datasets across all 5 assets: %s", SUPPORTED_ASSETS)

    for asset in SUPPORTED_ASSETS:
        ticker = asset.replace("_", ".").upper()
        if not ticker.endswith(".NS"):
            ticker = f"{ticker}.NS"
        try:
            downloader.download_market_data(symbol=ticker, start_date=start_date)
            results[asset] = True
            logger.info("Successfully refreshed parquet dataset for %s (%s)", asset, ticker)
        except Exception as err:
            logger.warning("Failed to refresh dataset for %s (%s): %s", asset, ticker, str(err))
            results[asset] = False

    return results