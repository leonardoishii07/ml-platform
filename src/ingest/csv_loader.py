import pandas as pd
from src.ingest.base import BaseDataLoader

class CSVLoader(BaseDataLoader):
    def __init__(self, path: str, **kwargs):
        self.path = path
        self.read_kwargs = kwargs

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path, **self.read_kwargs)
