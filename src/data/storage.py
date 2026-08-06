"""
Storage Engine for AlphaForge.

Handles:
- Saving datasets
- Loading datasets
- Metadata generation
- Metadata persistence
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StorageEngine:
    """
    Handles persistent storage of datasets.
    """

    def __init__(self, base_dir: Path = RAW_DATA_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_dir = self.base_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        logger.info("StorageEngine initialized at %s", self.base_dir)

    def _dataset_path(self, dataset_name: str) -> Path:
        dataset_name = dataset_name.lower()
        return self.base_dir / f"{dataset_name}.parquet"

    def save_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        overwrite: bool = False,
    ) -> Path:
        """
        Save dataset as Parquet.
        """

        if df.empty:
            raise ValueError("Cannot save an empty dataset.")

        path = self._dataset_path(dataset_name)

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Dataset '{dataset_name}' already exists."
            )

        logger.info("Saving dataset '%s'...", dataset_name)

        df.to_parquet(
            path,
            engine="pyarrow",
            index=True,
        )

        logger.info("Dataset saved successfully: %s", path)

        return path

    def load_dataset(
        self,
        dataset_name: str,
    ) -> pd.DataFrame:
        """
        Load a dataset.
        """

        path = self._dataset_path(dataset_name)

        if not path.exists():
            raise FileNotFoundError(path)

        logger.info("Loading dataset '%s'...", dataset_name)

        df = pd.read_parquet(
            path,
            engine="pyarrow",
        )

        logger.info("Loaded %d rows.", len(df))

        return df

    def dataset_exists(
        self,
        dataset_name: str,
    ) -> bool:

        return self._dataset_path(dataset_name).exists()

    def list_datasets(self) -> list[str]:

        return sorted(
            p.stem
            for p in self.base_dir.glob("*.parquet")
        )

    def generate_metadata(
        self,
        df: pd.DataFrame,
        dataset_name: str,
    ) -> dict[str, Any]:
        """
        Generate dataset metadata.
        """

        path = self._dataset_path(dataset_name)

        return {
            "dataset_name": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max()),
            "created_at": datetime.utcnow().isoformat(),
            "file_size_bytes": path.stat().st_size if path.exists() else None,
        }

    def save_metadata(
        self,
        dataset_name: str,
        metadata: dict[str, Any],
    ) -> Path:
        """
        Save metadata as JSON.
        """

        path = self.metadata_dir / f"{dataset_name.lower()}.json"

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

        logger.info("Metadata saved successfully: %s", path)

        return path