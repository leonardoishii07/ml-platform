import pandas as pd
from typing import List, Dict, Any
from src.utils.logger import get_logger
from .feature_selector_factory import get_feature_selector

logger = get_logger()

class FeatureSelectionRunner:
    """Runner class for feature selection pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature selection runner.
        
        Args:
            config: Configuration dictionary for feature selection
        """
        self.config = config
        self.selected_features: List[str] = []
        
    def run(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Run the feature selection pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            pd.DataFrame: DataFrame with selected features
        """
        if not self.config.get('enabled', False):
            logger.info("Feature selection is disabled. Skipping...")
            return df
            
        selected_features = set()
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Run each enabled feature selection method
        for method in ['statistical', 'correlation', 'importance']:
            if self.config.get(method, {}).get('enabled', False):
                logger.info(f"Running {method} feature selection...")
                try:
                    selector = get_feature_selector({
                        'method': method,
                        'params': self.config[method]['params']
                    })
                    selector.fit(X, y)
                    method_features = selector.get_selected_features()
                    selected_features.update(method_features)
                    logger.info(f"{method.capitalize()} selection chose {len(method_features)} features")
                except Exception as e:
                    logger.error(f"Error in {method} feature selection: {str(e)}")
                    continue
        
        # Always include target column
        selected_features.add(target_col)
        self.selected_features = sorted(list(selected_features))
        
        logger.info(f"Total selected features: {len(self.selected_features)}")
        logger.info(f"Selected features: {self.selected_features}")
        
        return df[self.selected_features]
    
    def get_selected_features(self) -> List[str]:
        """
        Get the list of selected features.
        
        Returns:
            List[str]: Names of selected features
        """
        return self.selected_features
