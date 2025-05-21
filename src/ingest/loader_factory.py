from src.ingest.csv_loader import CSVLoader
from src.ingest.base import BaseDataLoader

def get_loader(config: dict) -> BaseDataLoader:
    data_config = config.get("data", {})
    if not data_config:
        raise ValueError("No data configuration found in the provided config.")
    source_type = data_config.get("type")
    path = data_config.get("path")
    kwargs = data_config.get("params", {})

    if source_type == "csv":
        return CSVLoader(path, **kwargs)
    
    raise ValueError(f"Unsupported data source type: {source_type}")
