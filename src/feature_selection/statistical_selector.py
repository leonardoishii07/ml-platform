import pandas as pd
from typing import Optional, List
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from .base import BaseFeatureSelector

class StatisticalSelector(BaseFeatureSelector):
    """Feature selector based on statistical tests (ANOVA F-test, mutual information)."""
    
    def __init__(self, method: str = 'f_test', k: int = 10, problem_type: str = 'classification'):
        """
        Initialize the statistical feature selector.
        
        Args:
            method: Feature selection method ('f_test' or 'mutual_info')
            k: Number of features to select
            problem_type: Type of problem ('classification' or 'regression')
        """
        super().__init__({'method': method, 'k': k, 'problem_type': problem_type})
        self.method = method
        self.k = k
        self.problem_type = problem_type
        self.selector = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'StatisticalSelector':
        """
        Fit the selector to the data.
        
        Args:
            X: Features DataFrame
            y: Target variable
            
        Returns:
            self: Fitted selector
        """
        if self.method == 'f_test':
            score_func = f_classif if self.problem_type == 'classification' else f_regression
        elif self.method == 'mutual_info':
            score_func = mutual_info_classif if self.problem_type == 'classification' else mutual_info_regression
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        self.selector = SelectKBest(score_func=score_func, k=self.k)
        self.selector.fit(X, y)
        self.selected_features = list(X.columns[self.selector.get_support()])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by selecting only the chosen features.
        
        Args:
            X: Features DataFrame
            
        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        if self.selector is None:
            raise RuntimeError("Selector must be fitted before calling transform.")
        return X[self.selected_features]
