from typing import Dict, Any, Optional
from .base import BaseFeatureSelector
from .statistical_selector import StatisticalSelector
from .correlation_selector import CorrelationSelector
from .importance_selector import ImportanceSelector

def get_feature_selector(config: Dict[str, Any]) -> BaseFeatureSelector:
    """
    Factory function to create feature selectors based on configuration.
    
    Args:
        config: Configuration dictionary containing:
            - method: Feature selection method
            - params: Parameters for the specific selector
            
    Returns:
        BaseFeatureSelector: Configured feature selector instance
    
    Raises:
        ValueError: If method is not supported
    """
    method = config.get('method', 'statistical')
    params = config.get('params', {})
    
    if method == 'statistical':
        return StatisticalSelector(**params)
    elif method == 'correlation':
        return CorrelationSelector(**params)
    elif method == 'importance':
        return ImportanceSelector(**params)
    else:
        raise ValueError(f"Unsupported feature selection method: {method}")
