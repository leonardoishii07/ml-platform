from abc import ABC, abstractmethod
import pandas as pd

class BaseDataLoader(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass
