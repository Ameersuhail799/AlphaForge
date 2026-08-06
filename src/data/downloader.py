"""
AlphaForge Market Data Downloader

Coordinates the complete data ingestion pipeline.
"""

from __future__ import annotations

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

        dataset_name = symbol.replace(".", "_")

        # Download
        df = self.provider.download(
            symbol=symbol,
            start_date=start_date,
        )

        # Validate
        report = self.validator.validate(df)

        if not report.passed:
            logger.error("Validation failed.")
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