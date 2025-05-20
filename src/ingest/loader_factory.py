from src.ingest.csv_loader import CSVLoader
from src.ingest.base import BaseDataLoader

def get_loader(config: dict) -> BaseDataLoader:
    source_type = config.get("type")
    path = config.get("path")
    kwargs = config.get("params", {})

    if source_type == "csv":
        return CSVLoader(path, **kwargs)
    
    raise ValueError(f"Unsupported data source type: {source_type}")
