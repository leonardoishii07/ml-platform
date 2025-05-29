import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EncoderHandler:
    def __init__(self, enabled: bool, strategy: str = "onehot", columns: list = None):
        self.enabled = enabled
        self.strategy = strategy
        self.columns = columns
        self.encoder = None
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        if not self.enabled:
            logger.info("EncoderHandler is disabled, skipping fit.")
            return
        if not self.columns:
            self.columns = df.select_dtypes(include="object").columns.tolist()

        logger.info(f"Fitting encoder '{self.strategy}' on columns: {self.columns}")

        if self.strategy == "onehot":
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.encoder.fit(df[self.columns])
        elif self.strategy == "label":
            self.encoder = {}
            for col in self.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.encoder[col] = le
        else:
            raise ValueError(f"Unsupported encoding strategy: {self.strategy}")

        self.fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("You must call fit() before transform().")

        df_encoded = df.copy()

        if self.strategy == "onehot":
            encoded = self.encoder.transform(df_encoded[self.columns])
            encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.columns), index=df_encoded.index)
            df_encoded = df_encoded.drop(columns=self.columns)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        elif self.strategy == "label":
            for col in self.columns:
                df_encoded[col] = self.encoder[col].transform(df_encoded[col].astype(str))

        logger.info("Transformed data using encoder")
        return df_encoded
