import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MissingValueHandler(TransformerMixin):
    def __init__(self, enabled: bool = True, strategy: str = "mean", fill_value=None, knn_neighbors: int = 5, interpolation_method: str = "linear", direction: str = "forward", columns = None):
        self.enabled = enabled
        self.strategy = strategy
        self.fill_value = fill_value
        self.knn_neighbors = knn_neighbors
        self.interpolation_method = interpolation_method
        self.direction = direction
        self.fill_values_ = {}
        self.knn_imputer = None
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        if not self.enabled:
            logger.info("MissingValueHandler is disabled, skipping fit.")
            return self
        if self.columns is not None:
            X = X[self.columns]
        if self.strategy == "mean":
            self.fill_values_ = X.select_dtypes(include=["number"]).mean()
        elif self.strategy == "median":
            self.fill_values_ = X.select_dtypes(include=["number"]).median()
        elif self.strategy == "most_frequent":
            self.fill_values_ = X.mode().iloc[0]
        elif self.strategy == "constant":
            self.fill_values_ = {col: self.fill_value for col in X.columns}
        elif self.strategy == "knn":
            self.knn_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            self.knn_imputer.fit(X.select_dtypes(include=["number"]))
        elif self.strategy in ["interpolation", "fill_forward", "fill_backward"]:
            pass  # handled in transform
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        logger.info(f"Fitted MissingValueHandler with strategy={self.strategy}")
        return self

    def transform(self, X: pd.DataFrame):
        if self.strategy in ["mean", "median", "most_frequent", "constant"]:
            return X.fillna(self.fill_values_)

        elif self.strategy == "interpolation":
            return X.interpolate(method=self.interpolation_method, axis=0)

        elif self.strategy == "fill_forward":
            return X.fillna(method="ffill")

        elif self.strategy == "fill_backward":
            return X.fillna(method="bfill")

        elif self.strategy == "knn":
            X_num = X.select_dtypes(include=["number"])
            X_nonum = X.select_dtypes(exclude=["number"])
            X_num_imputed = pd.DataFrame(
                self.knn_imputer.transform(X_num),
                columns=X_num.columns,
                index=X.index
            )
            return pd.concat([X_num_imputed, X_nonum], axis=1)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
