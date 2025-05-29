# src/preprocessing/feature_engineering/auto_feature_generator.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

class AutoFeatureGenerator:
    def __init__(self, enabled: bool, target_col, problem_type="classification", correlation_threshold=0.05, top_k=10):
        self.enabled = enabled
        self.target_col = target_col
        self.problem_type = problem_type
        self.correlation_threshold = correlation_threshold
        self.top_k = top_k

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return df
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")
        if self.problem_type not in ["classification", "regression"]:
            raise ValueError("problem_type must be either 'classification' or 'regression'.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        target = df[self.target_col]
        features = df.drop(columns=[self.target_col])
        num_cols = features.select_dtypes(include=[np.number]).columns

        new_features = pd.DataFrame(index=df.index)
        scores = {}

        # Generate basic interaction features
        for i, col1 in enumerate(num_cols):
            for col2 in num_cols[i+1:]:
                interaction = features[col1] * features[col2]
                name = f"{col1}_x_{col2}"

                if self.problem_type == "classification":
                    score = mutual_info_classif(interaction.values.reshape(-1, 1), target, discrete_features=False)[0]
                else:
                    score = mutual_info_regression(interaction.values.reshape(-1, 1), target)[0]

                if score > self.correlation_threshold:
                    new_features[name] = interaction
                    scores[name] = score

        # Optionally keep top_k only
        if self.top_k:
            top_features = sorted(scores, key=scores.get, reverse=True)[:self.top_k]
            new_features = new_features[top_features]

        return pd.concat([df, new_features], axis=1)
