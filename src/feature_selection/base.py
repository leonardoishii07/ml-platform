from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional, Dict, Any

class BaseFeatureSelector(ABC):
    """Base class for feature selection methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.selected_features: Optional[List[str]] = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseFeatureSelector':
        """
        Fit the feature selector to the data.
        
        Args:
            X: Features DataFrame
            y: Target variable (optional)
            
        Returns:
            self: The fitted selector
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by selecting only the chosen features.
        
        Args:
            X: Features DataFrame
            
        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the selector and transform the data.
        
        Args:
            X: Features DataFrame
            y: Target variable (optional)
            
        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        return self.fit(X, y).transform(X)
    
    def get_selected_features(self) -> Optional[List[str]]:
        """
        Get the list of selected feature names.
        
        Returns:
            List[str]: Names of selected features
        """
        return self.selected_features
