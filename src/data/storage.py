"""Storage engine for AlphaForge market datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StorageEngine:
    """Manage persistent storage of AlphaForge datasets."""

    def __init__(self, base_dir: Path = RAW_DATA_DIR) -> None:
        """Initialize the storage engine.

        Args:
            base_dir: Directory used to store datasets.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        logger.info("StorageEngine initialized at %s", self.base_dir)

    def _get_path(self, dataset_name: str) -> Path:
        """Return the Parquet path for a dataset."""
        safe_name = dataset_name.strip().lower()

        if not safe_name:
            raise ValueError("Dataset name cannot be empty.")

        if any(char in safe_name for char in r'\/:*?"<>|'):
            raise ValueError("Dataset name contains invalid characters.")

        return self.base_dir / f"{safe_name}.parquet"

    def save_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        overwrite: bool = False,
    ) -> Path:
        """Save a DataFrame as a Parquet dataset.

        Args:
            df: Dataset to save.
            dataset_name: Dataset identifier.
            overwrite: Whether an existing dataset may be replaced.

        Returns:
            Path of the saved dataset.

        Raises:
            ValueError: If the DataFrame is empty.
            FileExistsError: If the dataset exists and overwrite is False.
            RuntimeError: If saving fails.
        """
        if df.empty:
            raise ValueError("Cannot save an empty dataset.")

        path = self._get_path(dataset_name)

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Dataset already exists: {path}. "
                "Set overwrite=True to replace it."
            )

        logger.info("Saving dataset '%s'...", dataset_name)

        try:
            df.to_parquet(path, engine="pyarrow", index=True)
        except Exception as error:
            logger.exception("Failed to save dataset '%s'.", dataset_name)
            raise RuntimeError(
                f"Unable to save dataset '{dataset_name}'."
            ) from error

        logger.info("Dataset saved successfully: %s", path)

        return path

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a stored Parquet dataset."""
        path = self._get_path(dataset_name)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        logger.info("Loading dataset '%s'...", dataset_name)

        try:
            df = pd.read_parquet(path, engine="pyarrow")
        except Exception as error:
            logger.exception("Failed to load dataset '%s'.", dataset_name)
            raise RuntimeError(
                f"Unable to load dataset '{dataset_name}'."
            ) from error

        logger.info("Loaded %d rows.", len(df))

        return df

    def dataset_exists(self, dataset_name: str) -> bool:
        """Check whether a dataset exists."""
        return self._get_path(dataset_name).exists()

    def list_datasets(self) -> list[str]:
        """Return available dataset names."""
        return sorted(path.stem for path in self.base_dir.glob("*.parquet"))

    def generate_metadata(
        self,
        df: pd.DataFrame,
        dataset_name: str,
    ) -> dict[str, Any]:
        """Generate metadata describing a dataset."""
        path = self._get_path(dataset_name)

        start_date = None
        end_date = None

        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            start_date = df.index.min().isoformat()
            end_date = df.index.max().isoformat()

        return {
            "dataset_name": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "start_date": start_date,
            "end_date": end_date,
            "save_timestamp": datetime.now(timezone.utc).isoformat(),
            "file_size_bytes": path.stat().st_size if path.exists() else None,
            "column_names": [str(column) for column in df.columns],
        }