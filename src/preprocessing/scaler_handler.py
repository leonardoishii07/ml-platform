import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ScalerHandler:
    def __init__(self, enabled: bool = True, strategy: str = "standard", columns: list = None):
        self.enabled = enabled
        self.strategy = strategy
        self.columns = columns
        self.scaler = None

    def fit(self, df: pd.DataFrame):
        if not self.enabled:
            logger.info("ScalerHandler is disabled, skipping fit.")
            return
        if not self.columns:
            self.columns = df.select_dtypes(include="number").columns.tolist()

        if self.strategy == "standard":
            self.scaler = StandardScaler()
        elif self.strategy == "minmax":
            self.scaler = MinMaxScaler()
        elif self.strategy == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler strategy: {self.strategy}")

        logger.info(f"Fitting scaler '{self.strategy}' on columns: {self.columns}")
        self.scaler.fit(df[self.columns])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise RuntimeError("You must call fit() before transform().")

        df_scaled = df.copy()
        df_scaled[self.columns] = self.scaler.transform(df[self.columns])
        logger.info("Transformed data using scaler")
        return df_scaled
