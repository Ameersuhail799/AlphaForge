from src.data.storage import StorageEngine


def load_test_data():
    storage = StorageEngine()
    return storage.load_dataset("RELIANCE_NS")