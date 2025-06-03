import pandas as pd
import numpy as np
from typing import Optional, List
from .base import BaseFeatureSelector

class CorrelationSelector(BaseFeatureSelector):
    """Feature selector based on correlation analysis."""
    
    def __init__(self, method: str = 'pearson', threshold: float = 0.95, target_corr_threshold: float = 0.01):
        """
        Initialize the correlation-based feature selector.
        
        Args:
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            threshold: Correlation threshold for removing highly correlated features
            target_corr_threshold: Minimum absolute correlation with target to keep feature
        """
        super().__init__({
            'method': method,
            'threshold': threshold,
            'target_corr_threshold': target_corr_threshold
        })
        self.method = method
        self.threshold = threshold
        self.target_corr_threshold = target_corr_threshold
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CorrelationSelector':
        """
        Fit the selector to the data.
        
        Args:
            X: Features DataFrame
            y: Target variable (optional)
            
        Returns:
            self: Fitted selector
        """
        # Calculate correlation matrix
        corr_matrix = X.corr(method=self.method)
        
        # Find highly correlated feature pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        
        # For each feature, find highly correlated pairs and keep the one with
        # higher correlation with target (if provided) or higher variance
        for feature in upper_tri.columns:
            correlated_features = upper_tri[feature][
                abs(upper_tri[feature]) > self.threshold
            ].index.tolist()
            
            if correlated_features:
                if y is not None:
                    # Keep feature with highest correlation with target
                    target_corrs = abs(pd.concat([X[correlated_features], X[feature]], axis=1)
                                    .apply(lambda x: x.corr(y)))
                    to_drop.update(
                        target_corrs[target_corrs != target_corrs.max()].index
                    )
                else:
                    # Keep feature with highest variance
                    variances = X[[feature] + correlated_features].var()
                    to_drop.update(
                        variances[variances != variances.max()].index
                    )
        
        # If target is provided, remove features with low correlation to target
        if y is not None:
            target_corrs = abs(X.corrwith(y, method=self.method))
            to_drop.update(
                target_corrs[target_corrs < self.target_corr_threshold].index
            )
        
        self.selected_features = [f for f in X.columns if f not in to_drop]
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
