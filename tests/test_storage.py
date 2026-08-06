from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data.storage import StorageEngine


def test_storage_engine():
    with TemporaryDirectory() as temp_dir:
        storage = StorageEngine(Path(temp_dir))

        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [103.0, 104.0],
                "Adj Close": [103.0, 104.0],
                "Volume": [10000, 12000],
            },
            index=pd.to_datetime(["2026-08-04", "2026-08-05"]),
        )

        path = storage.save_dataset(df, "test_market_data")

        assert path.exists()
        assert storage.dataset_exists("test_market_data")

        loaded_df = storage.load_dataset("test_market_data")

        pd.testing.assert_frame_equal(df, loaded_df)

        assert "test_market_data" in storage.list_datasets()

        metadata = storage.generate_metadata(
            loaded_df,
            "test_market_data",
        )

        assert metadata["rows"] == 2
        assert metadata["columns"] == 6
        assert metadata["file_size_bytes"] > 0

        print("\nStorage Engine: ALL TESTS PASSED")


if __name__ == "__main__":
    test_storage_engine()