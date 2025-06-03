import pandas as pd
from typing import Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base import BaseFeatureSelector

class ImportanceSelector(BaseFeatureSelector):
    """Feature selector based on feature importance from tree-based models."""
    
    def __init__(self, n_features: int = 10, problem_type: str = 'classification', 
                 importance_threshold: float = 0.01):
        """
        Initialize the importance-based feature selector.
        
        Args:
            n_features: Number of features to select
            problem_type: Type of problem ('classification' or 'regression')
            importance_threshold: Minimum importance threshold to keep feature
        """
        super().__init__({
            'n_features': n_features,
            'problem_type': problem_type,
            'importance_threshold': importance_threshold
        })
        self.n_features = n_features
        self.problem_type = problem_type
        self.importance_threshold = importance_threshold
        self.model = None
        self.feature_importances_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ImportanceSelector':
        """
        Fit the selector to the data.
        
        Args:
            X: Features DataFrame
            y: Target variable
            
        Returns:
            self: Fitted selector
        """
        # Initialize model based on problem type
        if self.problem_type == 'classification':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        # Fit model and get feature importances
        self.model.fit(X, y)
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        # Select features based on importance threshold and n_features
        selected = self.feature_importances_[
            self.feature_importances_ > self.importance_threshold
        ].head(self.n_features)
        
        self.selected_features = selected.index.tolist()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by selecting only the chosen features.
        
        Args:
            X: Features DataFrame
            
        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        if self.selected_features is None:
            raise RuntimeError("Selector must be fitted before calling transform.")
        return X[self.selected_features]
        
    def get_feature_importances(self) -> Optional[pd.Series]:
        """
        Get the feature importance scores.
        
        Returns:
            pd.Series: Feature importance scores
        """
        return self.feature_importances_
